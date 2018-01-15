'''
Bounding box filtering module for object detection algorithms

Copyright (C) <2018>  <Josue R. Cuevas>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''






'''
    Intersection over Union estimation between two boxes
'''
def batch_iou(boxes, box):
  """Compute the Intersection-Over-Union of a batch of boxes with another
  box.

  Args:
    box1: 2D array of [cx, cy, width, height].
    box2: a single array of [cx, cy, width, height]
  Returns:
    ious: array of a float number in range [0, 1].
  """
  lr = np.maximum(
      np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
      np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
      0
  )
  tb = np.maximum(
      np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
      np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
      0
  )
  inter = lr*tb
  union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
  return inter/union


'''
  In DBSCAN algorithm we dont use the usual euclidean distance, what we do actually is
  to determine how boxes overlap according to (Cx, Cy, H, W), this will give us an
  indication how close the boxes are.

  We dont consider overlapping boxes that belong to different class. Every box will compete with
  other candidates but as the Boxes are removed, this comparision reduces its computational demand.
'''
def region_query(candidate_boxes, sorted_probs):
  if len(sorted_probs) > 0:
    '''
      The trick here is to calculate only a upper-diagonal matrix instead of an NxN which is commonly
      the case.
      The upper-diagonal matrix will contain the overlapping values of each candidate box with other candidates
      in the following way:

        Overlap Similarity Matrix:
          candidate_1: [1-2, 1-3, 1-4, 1-5, 1-6, ..., 1-N]
          candidate_1: [---, 2-3, 2-4, 2-5, 2-6, ..., 2-N]
          candidate_1: [---, ---, 3-4, 3-5, 3-6, ..., 3-N]
          candidate_1: [---, ---, ---, 4-5, 4-6, ..., 4-N]
          candidate_1: [---, ---, ---, ---, 5-6, ..., 5-N]
          ... ...       ...       ...         ...     ...
          candidate_1: [---, ---, ---, ---, ---, ..., N-N]
    '''
    similarities = []
    for b_i in range(len(sorted_probs)-1):
          similarities.append(batch_iou(candidate_boxes[sorted_probs[b_i+1:]], candidate_boxes[sorted_probs[b_i]]))
          # print("%d -> Similarities: " % b_i)
          # print(np.array(similarities[len(similarities)-1]).shape)

    return similarities
  else:
    print("There are no candidate boxes, quitting...")
    return False


def expand_superCluster(class_indices, similarity_upper_matrix, threshold, survivors):
  '''
    Here we are basically comparing every candidate sorted in decreasing order by their
    probability values. The comparison is reduced every time a candidate is removed from
    the list of Bounding Boxes, where the similarity upper-diagonal matrix is used
  '''
  for i in range(len(class_indices)-1):
    for j in range(i+1, len(class_indices)-1):
      # print(class_indices[i], class_indices[j] - class_indices[i])

      overlap_value = similarity_upper_matrix[class_indices[i]][class_indices[j] - class_indices[i] - 1]
      if overlap_value > threshold:
        survivors[class_indices[j]] =  False








'''
    =======================================
    THIS IS THE ENTRY FUNCTION
    =======================================
'''
def filter_prediction(boxes, probs, cls_idx):
  """Filter bounding box predictions with probability threshold and
  non-maximum supression.

  Args:
    boxes: array of [cx, cy, w, h].
    probs: array of probabilities
    cls_idx: array of class indices
  Returns:
    final_boxes: array of filtered bounding boxes.
    final_probs: array of filtered probabilities
    final_cls_idx: array of filtered class indices
  """
  TOP_N_DETECTION = 64 # Max number of boxes
  NMS_THRESH = 0.4 # for Box to be filtered out
  CLASSES = 8
  PROB_THRESH = 0.005 # for class to be considered

  if TOP_N_DETECTION < len(probs) and TOP_N_DETECTION > 0:
    print("Using top %d (detected: %d) detections" % (TOP_N_DETECTION, len(probs)))
    order = probs.argsort()[:-TOP_N_DETECTION - 1:-1]
    probs = probs[order]
    boxes = boxes[order]
    cls_idx = cls_idx[order]
  else:
    print("Using top %d (detected: %d) detections" % (TOP_N_DETECTION, len(probs)))
    filtered_idx = np.nonzero(probs > PROB_THRESH)[0]
    probs = probs[filtered_idx]
    boxes = boxes[filtered_idx]
    cls_idx = cls_idx[filtered_idx]

  # for statistics
  statistic_val = 0.0
  for trials in range(1000):

    final_boxes = []
    final_probs = []
    final_cls_idx = []

    # print("================================")
    # print("Filtering boxes defined as follows:")
    # print(boxes.shape)
    # print(probs.shape)
    # print(cls_idx.shape)
    # print("================================")

    # ------------------------------------- DBSCAN
    start_nms = time.time()

    # probability orders
    order_list = probs.argsort()[::-1]
    # print(order_list)

    # candidates that will not be removed flag
    boxes_kept = [True]*len(order_list)

    similarities = region_query(boxes, order_list) #single time calculation

    '''
      We already know our Super-Cluster which is class-based, however inside each we may have
      many overlapping boxes which could be consider as sub-clusters inside each class, the point
      is to find all candidate boxes that overlap a candidate "k", these highly overlapping neighbors
      can be eliminated since it is assumed they belong to the same sub-cluster, or same bean prediction.

      Steps:
        1. Divide data points (Boxes) into super-clusters of classes. We may change this.
        2. From highest to lowest confidence, order the members of the super-cluster, and
           start to gather all neighbors to a particular candidate "k", sub-cluster neighbors according
           to overlapping distance upper-diagonal matrix. (It is already ordered so no need to re-order,
           we just have to be careful in the indexing)
        3. Eliminate all the members in that cluster that are highly overlapped and leave only the one
           with the highest confidence.
        4. Repeat step 2-3 until no more candidates are left.
    '''
    # SUPER-CLUSTER
    for c in range(CLASSES):
      # print("class: %d" % c)

      # take all points in that class and start cluster them inside the class
      idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]
      # print (idx_per_class)

      # cluster and removal of points
      expand_superCluster(idx_per_class, similarities, NMS_THRESH, boxes_kept)

      # keeping boxes in a list, we dont scan more than the points just used
      for i in range(len(idx_per_class)):
        if boxes_kept[idx_per_class[i]]:
          final_boxes.append(boxes[idx_per_class[i]])
          final_probs.append(probs[idx_per_class[i]])
          final_cls_idx.append(c)

    end_nms = time.time()

    statistic_val += ((end_nms-start_nms)*1000.0)
  print("&&&&&&&&&&&&\n NMS has taken %f ms in processing (avg) \n&&&&&&&&&&&&&&&" % (statistic_val / (trials)))
    # ------------------------------------- DBSCAN

  return final_boxes, final_probs, final_cls_idx
