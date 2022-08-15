def remove_redundant_info(dataset = None):
    '''
        "Trip Miles" is likely to provide as much information as:
            1. 'Pickup Centroid Latitude'
            2. 'Pickup Centroid Longitude'
            4. 'Dropoff Centroid Latitude'
            5. Dropoff Centroid Longitude
        
        Centroids "Latitude" and "Longitude" provide the same info as (just different format):
            1. 'Dropoff Centroid  Location
            2. 'Pickup Centroid Location'
        
        "Trip Seconds" is likely to provide as much information as:
            1, 'Trip Start Timestamp'
            2. 'Trip End Timestamp'
        
        'Trip Total' are a combination of 'Fare', 'Tips', 'Tolls', and 'Extras', therefore we 
        should not use this column otherwise the relationship will be pretty much linear
    '''
    new_dataset = dataset.drop(['Trip Start Timestamp', 'Trip End Timestamp',
                                'Pickup Centroid Latitude', 'Pickup Centroid Longitude', 
                                'Pickup Centroid Location', 'Dropoff Centroid Latitude', 
                                'Dropoff Centroid Longitude', 'Dropoff Centroid  Location',
                                'Trip Total'], axis=1)
    
    return new_dataset

def get_datasets_for_models(dataset_cleaned = None):
    pooled_inoutchicago_privacy_nonprivacy_data = dataset_cleaned.drop(['Pickup Census Tract', 'Dropoff Census Tract',
                                                'Pickup Community Area', 'Dropoff Community Area'
                                            ], axis=1)
    inchicago_noprivacy_data = dataset_cleaned[dataset_cleaned['Pickup Census Tract'].notna() &
                                                dataset_cleaned['Pickup Census Tract'].notnull() & 
                                                dataset_cleaned['Dropoff Census Tract'].notna() &
                                                dataset_cleaned['Dropoff Census Tract'].notnull() &
                                                dataset_cleaned['Pickup Community Area'].notna() &
                                                dataset_cleaned['Dropoff Community Area'].notnull()
                                            ]

    return pooled_inoutchicago_privacy_nonprivacy_data, inchicago_noprivacy_data