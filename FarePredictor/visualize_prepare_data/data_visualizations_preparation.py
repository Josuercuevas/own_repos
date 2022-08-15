from os.path import join
import pandas as pd
from helper_visualization import *
from helper_preparation import *

dataset_full = pd.read_csv(join('..', join('dataset', 'taxy_trips_full.csv')), sep=',')

# TARGET is Fare, so I will separate samples without fare from the ones that have
data_withFare, data_noFare = get_with_no_fare(dataset_full=dataset_full)

# this dataset has all Fares with valid data
dataset_all = data_withFare

outsideChicago, outsideChicago_privacy, outsideChicago_missing, outsideChicago_complete = get_outside_chicago_info(dataset_all=dataset_all)
print(f'Columns in Dataframe are: {dataset_all.columns.to_list()}')
print(f'Total number of datapoints are: {len(dataset_all)}')

print(f'The number of datapoints outside chicago are: {len(outsideChicago)}')
print(f'The number of datapoints outside chicago hidden because of privacy are: {len(outsideChicago_privacy)}')
print(f'The number of datapoints outside chicago missing are: {len(outsideChicago_missing)}')
print(f'The number of datapoints outside chicago with all information are: {len(outsideChicago_complete)}')


inside_chicago, privacy, missing, complete = get_inside_chicago_info(dataset_all=dataset_all)
print(f'The number of datapoints inside chicago are: {len(inside_chicago)}')
print(f'The number of datapoints inside chicago hidden because of privacy are: {len(privacy)}')
print(f'The number of datapoints inside chicago missing are: {len(missing)}')
print(f'The number of datapoints inside chicago with all information are: {len(complete)}')

# Draw proportion of Trips inside and Outside Chicago
get_piechart_inout_chicago(outsideChicago=outsideChicago, inside_chicago=inside_chicago)

# Draw proportion of Trips with Privacy filter activated vs deactivated
get_piechart_hidden_data(outsideChicago_privacy=outsideChicago_privacy, privacy=privacy, dataset_all=dataset_all)


# since true missing values represent a very small percentage, I will just remove them
print(f'Dataset after removing missing Fares and Features has: {len(dataset_all)} samples')

# plot Fares distributions before any type of cleaning
plot_fare_distribution_bfr_curation(dataset_all=dataset_all, inside_chicago=inside_chicago, outsideChicago=outsideChicago)

# quantiles to determine outliers for Columns with numerical values
q1 = dataset_all['Fare'].quantile(0.25)
q3 = dataset_all['Fare'].quantile(0.75)
iqr = q3 - q1
lower_bound = max(0, q1 - 1.5*iqr)
upper_bound = q3 + 1.5*iqr
print(f'Fares Quantiles information is: LB ({lower_bound}), UB ({upper_bound}), IQR ({q3}-{q1} = {iqr})')
if lower_bound < upper_bound:
    dataset_cleaned = dataset_all.drop(dataset_all[(dataset_all['Fare']<=lower_bound) | 
                                                        (dataset_all['Fare']>=upper_bound)
                                                    ].index)

# Remove NaN or Null as these trips make no sense
dataset_cleaned = dataset_cleaned[dataset_cleaned['Trip Miles'].notnull() & 
                                            dataset_cleaned['Trip Miles'].notna()
                                        ]
q1 = dataset_cleaned['Trip Miles'].quantile(0.25)
q3 = dataset_cleaned['Trip Miles'].quantile(0.75)
iqr = q3 - q1
lower_bound = max(0, q1 - 1.5*iqr)
upper_bound = q3 + 1.5*iqr
print(f'Trip Miles Quantiles information is: LB ({lower_bound}), UB ({upper_bound}), IQR ({q3}-{q1} = {iqr})')
if lower_bound < upper_bound:
    dataset_cleaned = dataset_cleaned.drop(dataset_cleaned[(dataset_cleaned['Trip Miles']<=lower_bound) | 
                                                                        (dataset_cleaned['Trip Miles']>=upper_bound)
                                                                    ].index)

# Remove NaN or Null as these trips make no sense
dataset_cleaned = dataset_cleaned[dataset_cleaned['Trip Seconds'].notnull() & 
                                            dataset_cleaned['Trip Seconds'].notna()
                                        ]
q1 = dataset_cleaned['Trip Seconds'].quantile(0.25)
q3 = dataset_cleaned['Trip Seconds'].quantile(0.75)
iqr = q3 - q1
lower_bound = max(0, q1 - 1.5*iqr)
upper_bound = q3 + 1.5*iqr
print(f'Trip Seconds Quantiles information is: LB ({lower_bound}), UB ({upper_bound}), IQR ({q3}-{q1} = {iqr})')
if lower_bound < upper_bound:
    dataset_cleaned = dataset_cleaned.drop(dataset_cleaned[(dataset_cleaned['Trip Seconds']<=lower_bound) | 
                                                                        (dataset_cleaned['Trip Seconds']>=upper_bound)
                                                                    ].index)


# Replacing Null and NaN by 0 as this means no tips, no IQR outlier removal as Fare should not depend on it
dataset_cleaned.loc[dataset_cleaned['Tips'].isnull() | dataset_cleaned['Tips'].isna(), 'Tips'] = 0

# Replacing Null and NaN by 0 as this means no tolls, no IQR outlier removal as Fare should not depend on it
dataset_cleaned.loc[dataset_cleaned['Tolls'].isnull() | dataset_cleaned['Tolls'].isna(), 'Tolls'] = 0

# Replacing Null and NaN by 0 as this means no extras, no IQR outlier removal as Fare should not depend on it
dataset_cleaned.loc[dataset_cleaned['Extras'].isnull() | dataset_cleaned['Extras'].isna(), 'Extras'] = 0

# Remove NaN or Null as these totals make no sense, no IQR outlier removal as Fare should not depend on it
dataset_cleaned = dataset_cleaned[dataset_cleaned['Trip Total'].notnull() & 
                                            dataset_cleaned['Trip Total'].notna()
                                        ]

# Remove NaN or Null as we don't know if Taxi ID affects the fare (car type info may be intrinsically encoded)
dataset_cleaned = dataset_cleaned[dataset_cleaned['Taxi ID'].notnull() & 
                                            dataset_cleaned['Taxi ID'].notna()
                                        ]

# Remove NaN or Null as we don't know if Payment Type affects the fare
dataset_cleaned = dataset_cleaned[dataset_cleaned['Payment Type'].notnull() & 
                                            dataset_cleaned['Payment Type'].notna()
                                        ]

# Remove NaN or Null as we don't know if Company affects the fare
dataset_cleaned = dataset_cleaned[dataset_cleaned['Company'].notnull() & 
                                            dataset_cleaned['Company'].notna()
                                        ]

# remove redundant information
dataset_cleaned = remove_redundant_info(dataset=dataset_cleaned)

# get fares proportions in/out chicago
print(f'Cleaned dataset left {len(dataset_cleaned)} samples')
plot_fare_distribution_afr_curation(dataset_all=dataset_cleaned)

# plot pie chart for data removed (missing, inconsistent data) from original dataset
get_piechart_cleaned_data(dataset_cleaned=dataset_cleaned, dataset_all=dataset_all)

''' 
    create 2 datasets:
        1. Only samples with all Features are present, meaning we drop samples with privacy and inside chicago
        2. DB with all pooled samples ignoring privacy and if is inside/outside chicago
'''
pooled_inoutchicago_privacy_nonprivacy_data, inchicago_noprivacy_data = get_datasets_for_models(dataset_cleaned=dataset_cleaned)

print(inchicago_noprivacy_data.columns.to_list())
ss = inchicago_noprivacy_data[inchicago_noprivacy_data.isnull().any(axis=1) | inchicago_noprivacy_data.isna().any(axis=1)]
print(f'Database with samples Inside Chicago and No Privacy activated: {len(inchicago_noprivacy_data)}')
print(f'Rows with weird values: {len(ss)}')

print(pooled_inoutchicago_privacy_nonprivacy_data.columns.to_list())
ss = pooled_inoutchicago_privacy_nonprivacy_data[pooled_inoutchicago_privacy_nonprivacy_data.isnull().any(axis=1) | pooled_inoutchicago_privacy_nonprivacy_data.isna().any(axis=1)]
print(f'Database combining In/Out Chicago and Private/Non-Private samples: {len(pooled_inoutchicago_privacy_nonprivacy_data)}')
print(f'Rows with weird values: {len(ss)}')

# write datasets to corresponding locations so we can train models later
filename = join(join(join(join('..'), 'dataset'), 'chicago_noprivacy'), 'dataset.csv')
inchicago_noprivacy_data.to_csv(filename, encoding='utf-8', index=False)
filename = join(join(join(join('..'), 'dataset'), 'pooled'), 'dataset.csv')
pooled_inoutchicago_privacy_nonprivacy_data.to_csv(filename, encoding='utf-8', index=False)
