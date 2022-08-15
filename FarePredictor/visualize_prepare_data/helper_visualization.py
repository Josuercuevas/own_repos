import matplotlib.pyplot as plt
import numpy as np

def get_with_no_fare(dataset_full = None):
    print(dataset_full.columns.to_list())
    data_withFare = dataset_full[dataset_full['Fare'].notnull() & dataset_full['Fare'].notna()]
    data_noFare = dataset_full[dataset_full['Fare'].isnull() | dataset_full['Fare'].isna()]

    print(f'Data WITH fares: {len(data_withFare)} / {len(dataset_full)}')
    print(f'Data WITHOUT fares: {len(data_noFare)} / {len(dataset_full)}')

    return data_withFare, data_noFare

def get_outside_chicago_info(dataset_all = None):
    # For sure outside chicago
    outsideChicago = dataset_all[dataset_all['Pickup Community Area'].isnull() | 
                                    dataset_all['Dropoff Community Area'].isnull()
                                ]
    # outside Chicago hidden because of privacy (not missing)
    outsideChicago_privacy = outsideChicago[outsideChicago['Pickup Census Tract'].isnull() &
                                                outsideChicago['Dropoff Census Tract'].isnull()
                                            ]
    # outside Chicago missing values in at least one of the Features
    outsideChicago_missing_valid = outsideChicago[outsideChicago['Pickup Census Tract'].notnull() |
                                                    outsideChicago['Dropoff Census Tract'].notnull()
                                                ]
    outsideChicago_missing = outsideChicago_missing_valid[outsideChicago_missing_valid.isnull().any(axis=1)]
    outsideChicago_complete = outsideChicago_missing_valid[outsideChicago_missing_valid.notnull().all(axis=1)]

    return outsideChicago, outsideChicago_privacy, outsideChicago_missing, outsideChicago_complete

def get_inside_chicago_info(dataset_all = None):
    # Hidden because of privacy (not missing)
    privacy_and_missing_valid = dataset_all[dataset_all['Pickup Community Area'].notnull() & 
                                                dataset_all['Dropoff Community Area'].notnull() &
                                                dataset_all['Pickup Community Area'].notna() & 
                                                dataset_all['Dropoff Community Area'].notna()
                                            ]
    privacy = privacy_and_missing_valid[privacy_and_missing_valid['Pickup Census Tract'].isnull() &
                                            privacy_and_missing_valid['Dropoff Census Tract'].isnull()
                                        ]
    # Missing values in at least one of the Features
    missing_valid = privacy_and_missing_valid[privacy_and_missing_valid['Pickup Census Tract'].notnull() |
                                                privacy_and_missing_valid['Dropoff Census Tract'].notnull() |
                                                privacy_and_missing_valid['Pickup Census Tract'].notna() |
                                                privacy_and_missing_valid['Dropoff Census Tract'].notna()
                                            ]
    missing = missing_valid[missing_valid.isnull().any(axis=1) | missing_valid.isna().any(axis=1)]
    complete = missing_valid[missing_valid.notnull().all(axis=1) | missing_valid.notna().all(axis=1)]

    return privacy_and_missing_valid, privacy, missing, complete

def plot_fare_distribution_bfr_curation(dataset_all = None, inside_chicago = None, outsideChicago = None):
    # now we need to check if the fares do make sense for the data OUTSIDE Chicago
    n_bins = len(dataset_all['Fare'].unique())
    FareDist = dataset_all['Fare'].hist(bins=n_bins)
    ax = FareDist
    ax.grid(False)
    ax.set_ylim([0, 100000])
    plt.xlabel("Fare values")
    plt.ylabel("Fare counts")
    plt.xticks(rotation=90)
    plt.title("Fare distribution for dataset WHOLE Chicago")
    plt.savefig('fare_whole_dataset_bfr_curation.png')
    plt.close('all')

    # now we need to check if the fares do make sense for data INSIDE Chicago
    n_bins = len(inside_chicago['Fare'].unique())
    FareDist = inside_chicago['Fare'].hist(bins=n_bins)
    ax = FareDist
    ax.grid(False)
    ax.set_ylim([0, 100000])
    plt.xlabel("Fare values")
    plt.ylabel("Fare counts")
    plt.xticks(rotation=90)
    plt.title("Fare distribution for dataset INSIDE Chicago")
    plt.savefig('fare_insideChicago_dataset_bfr_curation.png')
    plt.close('all')

    # now we need to check if the fares do make sense for data OUTSIDE Chicago
    n_bins = len(outsideChicago['Fare'].unique())
    FareDist = outsideChicago['Fare'].hist(bins=n_bins)
    ax = FareDist
    ax.grid(False)
    ax.set_ylim([0, 40000])
    plt.xlabel("Fare values")
    plt.ylabel("Fare counts")
    plt.xticks(rotation=90)
    plt.title("Fare distribution for dataset OUTSIDE Chicago")
    plt.savefig('fare_outsideChicago_dataset_bfr_curation.png')
    plt.close('all')

def plot_fare_distribution_afr_curation(dataset_all = None):
    # For sure outside chicago
    outsideChicago = dataset_all[dataset_all['Pickup Community Area'].isnull() | 
                                    dataset_all['Dropoff Community Area'].isnull()
                                ]
    inside_chicago = dataset_all[dataset_all['Pickup Community Area'].notnull() & 
                                                dataset_all['Dropoff Community Area'].notnull()
                                            ]
    
    # now we need to check if the fares do make sense for the data OUTSIDE Chicago
    n_bins = len(dataset_all['Fare'].unique())
    FareDist = dataset_all['Fare'].hist(bins=n_bins)
    ax = FareDist
    ax.grid(False)
    ax.set_ylim([0, 75000])
    plt.xlabel("Fare values")
    plt.ylabel("Fare counts")
    plt.xticks(rotation=90)
    plt.title("Fare distribution for dataset WHOLE Chicago")
    plt.savefig('fare_whole_dataset_afr_curation.png')
    plt.close('all')

    # now we need to check if the fares do make sense for data INSIDE Chicago
    n_bins = len(inside_chicago['Fare'].unique())
    FareDist = inside_chicago['Fare'].hist(bins=n_bins)
    ax = FareDist
    ax.grid(False)
    ax.set_ylim([0, 40000])
    plt.xlabel("Fare values")
    plt.ylabel("Fare counts")
    plt.xticks(rotation=90)
    plt.title("Fare distribution for dataset INSIDE Chicago")
    plt.savefig('fare_insideChicago_dataset_afr_curation.png')
    plt.close('all')

    # now we need to check if the fares do make sense for data OUTSIDE Chicago
    n_bins = len(outsideChicago['Fare'].unique())
    FareDist = outsideChicago['Fare'].hist(bins=n_bins)
    ax = FareDist
    ax.grid(False)
    ax.set_ylim([0, 5000])
    plt.xlabel("Fare values")
    plt.ylabel("Fare counts")
    plt.xticks(rotation=90)
    plt.title("Fare distribution for dataset OUTSIDE Chicago")
    plt.savefig('fare_outsideChicago_dataset_afr_curation.png')
    plt.close('all')

    data4boxplot = {
                    'outsideChicago': outsideChicago['Fare'],
                    'insideChicago': inside_chicago['Fare']
                }
    fig, ax = plt.subplots()
    ax.boxplot(data4boxplot.values(), sym='')
    ax.set_xticklabels(data4boxplot.keys())
    plt.ylabel("Fare")
    plt.title("Boxplot of Fares for rides inside/outside Chicago")
    plt.savefig('fares_in_out_chicago.png')
    plt.close('all')

    available_census_pick = dataset_all[dataset_all['Pickup Census Tract'].notna() & 
                                            dataset_all['Pickup Census Tract'].notnull()]
    available_census_drop = dataset_all[dataset_all['Dropoff Census Tract'].notna() &
                                            dataset_all['Dropoff Census Tract'].notnull()]
    
    n_bins = len(available_census_pick['Pickup Census Tract'].unique())
    pickup_census = available_census_pick['Pickup Census Tract'].hist(bins=n_bins)
    ax = pickup_census
    ax.grid(False)
    plt.xlabel("Pickup Census Tract values")
    plt.ylabel("Pickup Census Tract counts")
    plt.xticks(rotation=90)
    ax.set_ylim([0, 500])
    plt.title("Pickup Census Tract Data Distribution")
    plt.savefig('pickup_census_tract.png')
    plt.close('all')
    n_bins = len(available_census_drop['Dropoff Census Tract'].unique())
    dropoff_census = available_census_drop['Dropoff Census Tract'].hist(bins=n_bins)
    ax = dropoff_census
    ax.grid(False)
    plt.xlabel("Dropoff Census Tract values")
    plt.ylabel("Dropoff Census Tract counts")
    plt.xticks(rotation=90)
    ax.set_ylim([0, 500])
    plt.title("Dropoff Census Tract Data Distribution")
    plt.savefig('dropoff_census_tract.png')
    plt.close('all')

    available_community_pick = dataset_all[dataset_all['Pickup Community Area'].notna() & 
                                            dataset_all['Pickup Community Area'].notnull()]
    available_community_drop = dataset_all[dataset_all['Dropoff Community Area'].notna() &
                                            dataset_all['Dropoff Community Area'].notnull()]
    
    n_bins = len(available_community_pick['Pickup Community Area'].unique())
    pickup_community = available_community_pick['Pickup Community Area'].hist(bins=n_bins)
    ax = pickup_community
    ax.grid(False)
    plt.xlabel("Pickup Community Area values")
    plt.ylabel("Pickup Community Area counts")
    plt.xticks(rotation=90)
    plt.title("Pickup Community Area Data Distribution")
    plt.savefig('pickup_community_area.png')
    plt.close('all')
    n_bins = len(available_community_drop['Dropoff Community Area'].unique())
    dropoff_community = available_community_drop['Dropoff Community Area'].hist(bins=n_bins)
    ax = dropoff_community
    ax.grid(False)
    plt.xlabel("Dropoff Community Area values")
    plt.ylabel("Dropoff Community Area counts")
    plt.xticks(rotation=90)
    plt.title("Dropoff Community Area Data Distribution")
    plt.savefig('dropoff_community_area.png')
    plt.close('all')

    paymenttype = dataset_all[dataset_all['Payment Type'].notna() & dataset_all['Payment Type'].notnull()]
    n_bins = len(paymenttype['Payment Type'].unique())
    payment_type = paymenttype['Payment Type'].hist(bins=n_bins, density=True)
    ax = payment_type
    ax.grid(False)
    plt.xlabel("Payment Type values")
    plt.ylabel("Payment Type counts")
    plt.xticks(rotation=90)
    plt.title("Payment Type Data Distribution")
    plt.savefig('payment_type.png')
    plt.close('all')

    company = dataset_all[dataset_all['Company'].notna() & dataset_all['Company'].notnull()]
    n_bins = len(company['Company'].unique())
    company_type = company['Company'].hist(bins=n_bins, density=True)
    ax = company_type
    ax.grid(False)
    plt.xlabel("Company values")
    plt.ylabel("Company counts")
    plt.xticks(rotation=90)
    plt.title("Company Data Distribution")
    plt.savefig('company_type.png')
    plt.close('all')


def get_piechart_inout_chicago(outsideChicago = None, inside_chicago = None, 
                                    cash_outside = None, cash_inside = None):
    values = np.array([len(outsideChicago), len(inside_chicago)])
    labels = ['Outside Chicago', 'Inside Chicago']
    plt.pie(x=values, labels=labels, autopct='%1.1f%%')
    plt.title("Pie Chart of Trips Inside and Outside Chicago")
    plt.savefig('piechart_trips_in_out_chicago.png')
    plt.close('all')
    
    sumtotal = cash_outside+cash_inside
    plt.pie(x=np.array([round(cash_outside, 0), round(cash_inside, 0)]), labels=labels, 
                            autopct=lambda p:f'{round((p/100.0)*sumtotal,0):,} usd')
    plt.title("Pie Chart of Cash (USD) Inside and Outside Chicago")
    plt.savefig('piechart_cash_in_out_chicago.png')
    plt.close('all')

def get_piechart_hidden_data(outsideChicago_privacy = None, privacy = None, dataset_all = None):
    hidden_amount = len(outsideChicago_privacy)+len(privacy)
    not_hidden = len(dataset_all) - hidden_amount
    values = np.array([hidden_amount, not_hidden])
    labels = ['Privacy Filter Activated', 'Privacy Filter Inactive']
    plt.pie(x=values, labels=labels, autopct='%1.1f%%')
    plt.title("Pie Chart of Trips with Privacy Filter active/inactive")
    plt.savefig('piechart_trips_privacy_active_inactive.png')
    plt.close('all')

def get_piechart_cleaned_data(dataset_cleaned = None, dataset_all = None):
    values = np.array([len(dataset_cleaned), len(dataset_all)-len(dataset_cleaned)])
    labels = ['Cleaned Data', 'Removed Data']
    plt.pie(x=values, labels=labels, autopct='%1.1f%%')
    plt.title("Pie Chart of Proportion of data removed from Dataset")
    plt.savefig('piechart_removed_data.png')
    plt.close('all')

