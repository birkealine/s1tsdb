import ee

S1_PERIOD = 12  # 12 days between each S1 image for a given location and orbit


def manual_stats_from_s1(s1: ee.ImageCollection, start_date, n_dates: int = 32, n_days_in_slice: int = 4):
    """Compute stats from S1 collection"""

    # make sure the collection is limited to the first n_dates
    # end_date = start_date.advance(S1_PERIOD*(n_dates+1)-1, 'day')
    # s1 = s1.filterDate(start_date, end_date)

    # Compute stats
    stats_reducers = (
        ee.Reducer.mean()
        .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.median(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.max(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.min(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.skew(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.kurtosis(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.variance(), sharedInputs=True)
    )
    stats = s1.reduce(stats_reducers)
    vv_ptp = stats.select("VV_max").subtract(stats.select("VV_min")).rename("VV_ptp")
    vh_ptp = stats.select("VH_max").subtract(stats.select("VH_min")).rename("VH_ptp")
    stats = stats.addBands([vv_ptp, vh_ptp])

    # # Mean and std per slice (every 4 dates)
    # slices = s1.toList(n_dates)
    # for i in range(int(n_dates / 4)):
    #     slice = ee.ImageCollection.fromImages(slices.slice(i * 4, i * 4 + 4))
    #     slice_mean = slice.reduce(ee.Reducer.mean()).rename([f"VV_mean_slice{i}", f"VH_mean_slice{i}"])
    #     slice_std = slice.reduce(ee.Reducer.stdDev()).rename([f"VV_stdDev_slice{i}", f"VH_stdDev_slice{i}"])
    #     stats = stats.addBands(slice_mean).addBands(slice_std)

    # Temporal stats (mean and std)
    def process_slice(index):
        index = ee.Number(index)
        start_slice_date = ee.Date(start_date).advance(index.multiply(S1_PERIOD * n_days_in_slice), "day")
        end_slice_date = start_slice_date.advance(S1_PERIOD * n_days_in_slice, "day")
        slice = s1.filterDate(start_slice_date, end_slice_date)

        # need this weird syntax to get the name of the band server-side...
        vv_mean_name = ee.String("VV_mean_slice").cat(index.format("%d"))
        vh_mean_name = ee.String("VH_mean_slice").cat(index.format("%d"))
        vv_std_name = ee.String("VV_stdDev_slice").cat(index.format("%d"))
        vh_std_name = ee.String("VH_stdDev_slice").cat(index.format("%d"))

        slice_mean = slice.reduce(ee.Reducer.mean()).rename([vv_mean_name, vh_mean_name])
        slice_std = slice.reduce(ee.Reducer.stdDev()).rename([vv_std_name, vh_std_name])
        return ee.Image([slice_mean, slice_std])

    # Not working since toBands adds a weird prefix at the band names...
    num_slices = n_dates / n_days_in_slice
    slice_indices = ee.List.sequence(0, num_slices - 1)  # -1 since sequence(0,4) gives [0,1,2,3,4], not [0,1,2,3]
    stats_slices = ee.ImageCollection(slice_indices.map(process_slice)).toBands()
    names_without_prefix = []
    for i in range(int(num_slices)):
        names_without_prefix.append(f"VV_mean_slice{i}")
        names_without_prefix.append(f"VH_mean_slice{i}")
        names_without_prefix.append(f"VV_stdDev_slice{i}")
        names_without_prefix.append(f"VH_stdDev_slice{i}")
    stats_slices = stats_slices.rename(names_without_prefix)
    stats = stats.addBands(stats_slices)

    # This works but probably slower
    # def mergeImages(index, previous):
    #     return ee.Image(previous).addBands(process_slice(index))

    # num_slices = n_dates / n_days_in_slice
    # slice_indices = ee.List.sequence(1, num_slices - 1)
    # stats_slices = slice_indices.iterate(mergeImages, process_slice(0))
    # stats = stats.addBands(stats_slices)
    return stats

    # Assuming s1 is your ImageCollection sorted by time

    # # Define the number of images per time window
    # window_size = 4

    # # Function to create a composite for a given time window and calculate statistics
    # def create_composite_and_stats(start_index):
    #     # Create composite for the current time window
    #     end_index = start_index.add(window_size)
    #     time_window_collection = ee.ImageCollection(s1.limit(32).toList(end_index).slice(start_index, end_index))

    #     # Calculate statistics for this composite
    #     mean_composite = time_window_collection.reduce(ee.Reducer.mean()).rename(
    #         [f"VV_mean_{start_index.getInfo()}", f"VH_mean_{start_index.getInfo()}"]
    #     )
    #     std_composite = time_window_collection.reduce(ee.Reducer.stdDev()).rename(
    #         [f"VV_stdDev_{start_index.getInfo()}", f"VH_stdDev_{start_index.getInfo()}"]
    #     )

    #     # Combine mean and standard deviation
    #     return mean_composite.addBands(std_composite)

    # # Number of images in the collection
    # n_images = 32

    # # Generate a list of start indexes for each time window
    # start_indexes = ee.List.sequence(0, n_images - window_size, window_size)

    # # Map the function over each start index to create the composites
    # stats_composites = start_indexes.map(lambda index: create_composite_and_stats(ee.Number(index)))

    # # Convert the list of composites back to an ImageCollection
    # stats_image_collection = ee.ImageCollection.fromImages(stats_composites)
    # return stats_image_collection
