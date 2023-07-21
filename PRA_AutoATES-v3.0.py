import numpy as np
import rasterio
from osgeo import gdal
import os
from numpy.lib.stride_tricks import as_strided
from collections import deque
import sys
from datetime import datetime

def create_log_file():
    """
    Create a log file for the PRA process.

    Returns:
        file: A file object representing the log file in write mode.
    """
    # Define the path for the log file, located in the "PRA" directory, named "log.txt"
    log_file_path = os.path.join(os.getcwd(), "PRA", "log.txt")
    
    # Open the log file in write mode ('w+'), creating it if it doesn't exist
    return open(log_file_path, "w+")

def log_start_time(log_file):
    """
    Log the start time of the PRA process in the provided log file.

    Parameters:
        log_file (file): A file object representing the log file in write mode.

    """
    # Get the current date and time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    # Write the start time to the log file
    log_file.write("Start time = {}\n".format(current_time))

def log_parameters(log_file, forest_type, DEM, FOREST, radius, prob, winddir, windtol, pra_thd, sf):
    """
    Log the input parameters of the PRA process in the provided log file.

    Parameters:
        log_file (file): A file object representing the log file in write mode.
        forest_type (str): The type of forest ('pcc', 'stems', 'bav', or 'no_forest').
        DEM (str): The path to the Digital Elevation Model (DEM) file.
        FOREST (str): The path to the forest data file (not applicable for 'no_forest').
        radius (int): The radius of the circular sector for windshelter calculation.
        prob (float): The probability for quantile-based windshelter calculation.
        winddir (int): The wind direction in degrees (0-360).
        windtol (int): The wind tolerance angle in degrees.
        pra_thd (float): The threshold value for reclassifying PRA.
        sf (int): The size threshold in pixels for removing islands from the PRA.

    """
    # Format and write the input parameters to the log file
    log_file.write("forest_type: {}, DEM: {}, FOREST: {}, radius: {}, prob: {}, winddir: {}, windtol {}, pra_thd: {}, sf: {}\n".format(
        forest_type, DEM, FOREST, radius, prob, winddir, windtol, pra_thd, sf))

def create_path_if_not_exists(path):
    """
    Create the given path if it does not exist.

    Parameters:
        path (str): The path to be created if it does not exist.
        
    """
    # Create the directory at the specified path if it does not already exist
    # The 'exist_ok=True' argument ensures that the function does not raise an error if the directory already exists.
    os.makedirs(path, exist_ok=True)

def read_dem_raster(DEM):
    """
    Read a Digital Elevation Model (DEM) raster file and preprocess the data.

    Parameters:
        DEM (str): Path to the DEM raster file.

    Returns:
        numpy.ndarray: The 2D array representing the DEM data.
        affine.Affine: The transformation defining the pixel size and location.
        rasterio.crs.CRS: The coordinate reference system (CRS) of the raster.
    """
    # Open the DEM raster using rasterio
    with rasterio.open(DEM) as src:
        # Read the elevation values from band 1
        dem_data = src.read(1)

        # Get the profile of the raster dataset
        profile = src.profile

    # Replace invalid values (e.g., values less than -100) with nodata value (-9999)
    dem_data[np.where(dem_data < -100)] = -9999

    # Retrieve the transformation and CRS from the profile
    transform = profile['transform']
    crs = profile['crs']

    return dem_data, transform, crs

def calculate_slope(dem_data, transform):
    """
    Calculate the slope angle from a digital elevation model (DEM).

    Parameters:
        dem_data (numpy.ndarray): The 2D array representing the DEM.
        transform (affine.Affine): The transformation defining the pixel size and location.

    Returns:
        numpy.ndarray: The slope angle in degrees.
    """
    # Calculate the gradient in two dimensions using numpy gradient
    px, py = np.gradient(dem_data, *transform[0:2])

    # Calculate the slope angle using the gradient components
    slope_rad = np.arctan(np.sqrt(px ** 2 + py ** 2))

    # Convert the slope angle to degrees
    slope_deg = np.degrees(slope_rad)

    return slope_deg

def calculate_windshelter(dem_data, transform, radius, prob, winddir, windtol, cell_size):
    """
    Calculate windshelter values for a given DEM and wind direction.

    Parameters:
        dem_data (numpy.ndarray): The 2D array containing the DEM data.
        transform (affine.Affine): The transformation defining the pixel size and location.
        radius (int): The radius of the circular sector for windshelter calculation.
        prob (float): The probability for quantile-based windshelter calculation.
        winddir (int): The wind direction in degrees (0-360).
        windtol (int): The wind tolerance angle in degrees.
        cell_size (float): The cell size (pixel resolution) of the DEM in meters.

    Returns:
        numpy.ndarray: The 2D array containing windshelter values for the given DEM.
    """
    
    def sliding_window_view(arr, window_shape, steps):
        """
        Produce a view from a sliding, striding window over `arr`.
        The window is only placed in 'valid' positions - no overlapping over the boundary.

        Parameters:
            arr (numpy.ndarray): The input array to create the sliding window view.
            window_shape (tuple): The shape of the sliding window, e.g., (Wx, Wy, ...).
            steps (tuple): The step size for moving the window, e.g., (Sx, Sy, ...).

        Returns:
            numpy.ndarray: The sliding window view of the input array.
        """

        # Determine the shape of the input array along the sliding window axes (x, y, ...)
        in_shape = np.array(arr.shape[-len(steps):])  # [x, (...), z]

        # Convert the window shape and steps to numpy arrays for easy manipulation
        window_shape = np.array(window_shape)  # [Wx, (...), Wz]
        steps = np.array(steps)  # [Sx, (...), Sz]

        # Determine the size (in bytes) of an element in the input array
        nbytes = arr.strides[-1]

        # Calculate the number of per-byte steps to take to fill the window
        window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)

        # Calculate the number of per-byte steps to take to place the window
        step_strides = tuple(window_strides[-len(steps):] * steps)

        # Calculate the number of bytes to step to populate the sliding window view
        strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

        # Calculate the output shape of the sliding window view
        outshape = tuple((in_shape - window_shape) // steps + 1)  # ([X, (...), Z], ..., [Wx, (...), Wz])
        outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)

        # Create and return the sliding window view using the 'as_strided' function from NumPy
        return as_strided(arr, shape=outshape, strides=strides, writeable=False)

    def sector_mask(shape, centre, radius, angle_range):
        """
        Return a boolean mask for a circular sector. The start/stop angles in `angle_range` should be given in clockwise order.

        Parameters:
            shape (tuple): The shape of the 2D grid for which the mask is created, e.g., (rows, columns).
            centre (tuple): The center coordinates of the circular sector, e.g., (cx, cy).
            radius (float): The radius of the circular sector.
            angle_range (tuple): The start and stop angles of the sector in degrees, e.g., (start_angle, stop_angle).

        Returns:
            numpy.ndarray: A boolean mask representing the circular sector.
        """
            
        # Create a meshgrid of row and column indices (x, y)
        x, y = np.ogrid[:shape[0], :shape[1]]

        # Extract the center coordinates (cx, cy) from the input tuple
        cx, cy = centre

        # Convert the start and stop angles from degrees to radians
        tmin, tmax = np.deg2rad(angle_range)

        # Ensure that the stop angle is greater than the start angle (in radians)
        if tmax < tmin:
            tmax += 2 * np.pi

        # Convert the Cartesian coordinates to polar coordinates
        r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
        theta = np.arctan2(x - cx, y - cy) - tmin

        # Wrap angles between 0 and 2*pi to ensure they fall within the circular sector
        theta %= (2 * np.pi)

        # Create a circular mask based on whether each point lies within the specified radius
        circmask = r2 <= radius * radius

        # Create an angular mask based on whether each point falls within the specified angle range
        anglemask = theta <= (tmax - tmin)

        # Combine the circular and angular masks using element-wise multiplication
        # This will result in a boolean mask, where True indicates the points within the circular sector.
        a = circmask * anglemask

        return a

    def windshelter_prep(radius, direction, tolerance, cellsize):
        """
        Prepare data for windshelter calculation.

        Parameters:
            radius (int): The radius of the circular sector for windshelter calculation.
            direction (int): The wind direction in degrees (0-360).
            tolerance (int): The wind tolerance angle in degrees.
            cellsize (float): The cell size (pixel resolution) of the data in meters.

        Returns:
            numpy.ndarray: The 2D array representing the distance from the center to each cell.
            numpy.ndarray: The boolean mask representing the circular sector.
        """

        # Determine the size of the x and y dimensions of the matrix
        x_size = y_size = 2 * radius + 1

        # Create a meshgrid of x and y indices
        x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]

        # Calculate the coordinates of the cell center
        cell_center = (radius, radius)

        # Calculate the distance from the cell center to each cell using the Euclidean distance formula
        dist = (np.sqrt((x_arr - cell_center[0]) ** 2 + (y_arr - cell_center[1]) ** 2)) * cellsize

        # Calculate the boolean mask representing the circular sector using the sector_mask function
        mask = sector_mask(dist.shape, (radius, radius), radius, (direction, tolerance))

        # Correct a bug in the mask where the center cell is not included in the circular sector
        mask[radius, radius] = True

        return dist, mask

    def windshelter(x, prob, dist, mask, radius):
        """
        Calculate windshelter values for a circular window around each cell.

        Parameters:
            dem_data (numpy.ndarray): The 2D array containing the DEM data.
            transform (affine.Affine): The transformation defining the pixel size and location.
            radius (int): The radius of the circular sector for windshelter calculation.
            prob (float): The probability for quantile-based windshelter calculation.

        Returns:
            numpy.ndarray: The 3D array containing windshelter values for each cell in the window.
        """

        # Apply the circular sector mask to the input data
        data = x * mask

        # Set nodata and zero values in the data to NaN
        data[data == profile['nodata']] = np.nan
        data[data == 0] = np.nan

        # Get the value at the center cell
        center = data[radius, radius]

        # Set the center cell value to NaN to avoid its influence on the calculation
        data[radius, radius] = np.nan

        # Calculate the windshelter value using the arctan function and the distance matrix
        data = np.arctan((data - center) / dist)

        # Calculate the quantile-based windshelter value using the given probability
        data = np.nanquantile(data, prob)

        return data

    def windshelter_window(dem_data, transform, radius, prob):
        """
        Calculate windshelter values for a circular window around each cell.

        Parameters:
            dem_data (numpy.ndarray): The 2D array containing the DEM data.
            transform (affine.Affine): The transformation defining the pixel size and location.
            radius (int): The radius of the circular sector for windshelter calculation.
            prob (float): The probability for quantile-based windshelter calculation.

        Returns:
            numpy.ndarray: The 3D array containing windshelter values for each cell in the window.
        """
        # Get the distance matrix and circular sector mask using windshelter_prep
        dist, mask = windshelter_prep(radius, winddir - windtol + 270, winddir + windtol + 270, cell_size)

        # Create a sliding window view of the input data (dem_data)
        window = sliding_window_view(dem_data, ((radius * 2) + 1, (radius * 2) + 1), (1, 1))
        nc = window.shape[0]
        nr = window.shape[1]

        # Initialize a deque to store windshelter values for each cell in the window
        ws = deque()

        # Iterate through each window and calculate windshelter value for each cell
        for i in range(nc):
            for j in range(nr):
                data = window[i, j]

                # Calculate the windshelter value for the current cell using windshelter function
                data = windshelter(data, prob, dist, mask, radius).tolist()

                # Append the windshelter value to the deque
                ws.append(data)

        # Convert the deque to a numpy array and reshape it to match the input data dimensions
        data = np.array(ws)
        data = data.reshape(nc, nr)

        # Pad the data with nodata values to handle boundary effects
        data = np.pad(data, pad_width=radius, mode='constant', constant_values=-9999)

        # Reshape the data to add the channel dimension (1) and convert to float32 data type
        data = data.reshape(1, data.shape[0], data.shape[1])
        data = data.astype('float32')

        return data

    # Call the windshelter_window function with dem_data and transform as inputs
    data = windshelter_window(dem_data, transform, radius, prob)

    # Replace any NaN values in the windshelter data with -9999
    data = np.nan_to_num(data, nan=-9999)

    # Return the windshelter data
    return data

def calculate_ruggedness(dem_data, cell_size, target_cell_size, window):
    """
    Calculate the vector ruggedness of a Digital Elevation Model (DEM) using the Sappington method
    over larger scales than the cell size by resampling the DEM.

    Parameters:
        dem_data (numpy array): NumPy array representing the Digital Elevation Model (DEM) data.
        cell_size (float): The cell size of the original DEM in the same units as the elevation data.
        target_cell_size (float): The desired cell size for the resampled DEM in the same units as the elevation data.

    Returns:
        ruggedness_raster (numpy array): NumPy array representing the vector ruggedness values.
    """
    # Calculate the scaling factor for resampling
    scaling_factor = cell_size / target_cell_size

    # Resample the DEM to the target cell size
    target_shape = tuple(int(d * scaling_factor) for d in dem_data.shape[::-1])
    dem_data_resampled = resize(dem_data, target_shape, mode='reflect', anti_aliasing=True)

    # Calculate the slope and aspect from the resampled DEM data
    gradient_x, gradient_y = np.gradient(dem_data_resampled, target_cell_size, target_cell_size)
    slope_rad = np.arctan(np.sqrt(gradient_x ** 2 + gradient_y ** 2))
    aspect_rad = np.arctan2(gradient_y, -gradient_x) % (2 * np.pi)

    # Convert to degrees
    slope_deg = np.degrees(slope_rad)
    aspect_deg = np.degrees(aspect_rad)

    # Calculate xyz components
    slope_rad = np.radians(slope_deg)
    aspect_rad = np.radians(aspect_deg)
    xy_raster = np.sin(slope_rad)
    z_raster = np.cos(slope_rad)
    x_raster = np.sin(aspect_rad) * xy_raster
    y_raster = np.cos(aspect_rad) * xy_raster

    # Define the focal window for convolution
    focal_window = np.ones((window, window))

    # Perform convolution to calculate sums
    xsum_raster = convolve(x_raster, focal_window, mode='constant', cval=0.0)
    ysum_raster = convolve(y_raster, focal_window, mode='constant', cval=0.0)
    zsum_raster = convolve(z_raster, focal_window, mode='constant', cval=0.0)

    # Calculate vector ruggedness raster
    ruggedness_raster = 1 - np.sqrt(xsum_raster**2 + ysum_raster**2 + zsum_raster**2) / 9

    return ruggedness_raster

def cauchy_slope_function(slope_deg):
    """Calculate the Cauchy function for slope."""  
    a = 11
    b = 4
    c = 43
    f.write("Cauchy slope function: a={}, b={}, c={}\n".format(a, b, c))
    slopeC = 1/(1+((slope_deg-c)/a)**(2*b))
    slopeC = np.round(slopeC, 5)
    return slopeC

def cauchy_windshelter_function(windshelter):
    """Calculate the Cauchy function for windshelter."""
       # --- Define bell curve parameters for windshelter
    a = 3
    b = 10
    c = 3
    windshelterC = 1/(1+((windshelter-c)/a)**(2*b))
    windshelterC = np.round(windshelterC, 5)
    return windshelterC

def cauchy_ruggedness_function(vector_rougness):
    """Calculate the Cauchy function for terrain_ruggedness."""
    # --- Define bell curve parameters for vector roughness
    a = 0.01
    b = 5
    c = 0.007
    f.write("Cauchy vector roughness function: a={}, b={}, c={}\n".format(a, b, c))
    # Calculate the Cauchy function for vector roughness
    vector_roughnessC = 1 / (1 + ((vector_roughness - c) / a) ** (2 * b))
    vector_roughnessC = np.round(vector_roughnessC, 5)
    return vector_roughnessC 

def cauchy_forest_function(forest, forest_type):
    """Calculate the Cauchy function for the given forest type."""
    if forest_type in ['stems']:
        a = 350
        b = 2
        c = -120
        f.write("Cauchy forest function (stems): a={}, b={}, c={}\n".format(a, b, c))
    if forest_type in ['bav']:
        a = 20
        b = 3.5
        c = -10
        f.write("Cauchy forest function (bav): a={}, b={}, c={}\n".format(a, b, c))
    if forest_type in ['sen2cc']:
        a = 50 # still finalizing defualts for Sen2cc, likeily will be region dependent based on local forest structure
        b = 1.5
        c = 0
        f.write("Cauchy forest function (sen2cc): a={}, b={}, c={}\n".format(a, b, c)) 
    # --- Define bell curve parameters for percent canopy cover
    if forest_type in ['pcc', 'no_forest']:
        a = 40
        b = 3.5
        c = -15
        if forest_type in ['pcc']:
            f.write("Cauchy forest function (pcc): a={}, b={}, c={}\n".format(a, b, c))
        if forest_type in ['no_forest']:
            f.write("No forest input given\n")
    if forest_type in ['pcc', 'stems']:
        with rasterio.open(FOREST) as src:
            forest = src.read()
    if forest_type in ['no_forest']:
        with rasterio.open(DEM) as src:
            forest = src.read()
            forest = np.where(forest > -100, 0, forest)
    forestC = 1/(1+((forest-c)/a)**(2*b))
    # --- Ares with no forest and assigned -9999 will get a really small value which suggest dense forest. This function fixes this, but might have to be adjusted depending on the input dataset.
    forestC[np.where(forestC <= 0.00001)] = 1
    forestC = np.round(forestC, 5)
    return forestC

def save_raster_as_geotiff(PRA, transform, crs, filename):
    #Create the profile dictionary for the output GeoTIFF file
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',  # Data type of the PRA data (assuming float32 for continuous values)
        'nodata': -9999,     # Nodata value for the raster
        'width': PRA.shape[1],   # Number of columns in the raster
        'height': PRA.shape[0],  # Number of rows in the raster
        'count': 1,  # Number of bands in the raster
        'crs': crs,  # Coordinate reference system of the raster
        'transform': transform,  # Transformation defining the pixel size and location
    }

    # Save the PRA data to the GeoTIFF file
    with rasterio.open(filename, "w", **profile) as dest:
        dest.write(PRA.astype('float32'))
        
def reclassify_PRA(PRA, pra_thd):
    """
    Reclassify PRA to be used as input for FlowPy.

    Parameters:
        PRA (numpy.ndarray): The 2D array containing the PRA data.
        pra_thd (float): The threshold value for reclassification.

    Returns:
        numpy.ndarray: The reclassified PRA data.
    """
    # Set nodata value in raster metadata
    src.profile.update({'nodata': -9999})

    # Calculate the threshold value for reclassification
    pra_thd = pra_thd * 100

    # Reclassify PRA based on the threshold value
    PRA[np.where((0 <= PRA) & (PRA < pra_thd))] = 0
    PRA[np.where((pra_thd < PRA) & (PRA <= 100))] = 1

    return PRA

def remove_islands(sf): 
    sievefilter = sf + 1
    Image = gdal.Open('PRA/PRA_binary.tif', 1)  # open image in read-write mode
    Band = Image.GetRasterBand(1)
    gdal.SieveFilter(srcBand=Band, maskBand=None, dstBand=Band, threshold=sievefilter, connectedness=8, callback=gdal.TermProgress_nocb)
    del Image, Band  # close the datasets.

def log_stop_time(log_file):
    print('PRA complete')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    f.write("Stop time = {}\n".format(current_time))
    f.close()

def fuzzy_logic_operator(slopeC, windshelterC, forestC):
    """Perform the fuzzy logic operator."""
    print("Starting the Fuzzy Logic Operator")

    minvar = np.minimum(slopeC, windshelterC)
    minvar = np.minimum(minvar, forestC)

    PRA = (1-minvar)*minvar+minvar*(slopeC+windshelterC+forestC)/3
    PRA = np.round(PRA, 5)
    PRA = PRA * 100
    return PRA

def PRA(forest_type, DEM, FOREST, radius, prob, winddir, windtol, pra_thd, sf):
    # --- Create log file
    log_file = create_log_file()
    try:
        ##########################
        # --- Check input files
        ##########################
        path = os.path.join(os.getcwd(), "PRA")
        create_path_if_not_exists(path)
        log_start_time(log_file)
        # Check if path exits
        if not os.path.exists(DEM):
            print("The DEM path {} does not exist".format(DEM))
            return
        if forest_type in ['pcc', 'stems', 'bav', 'sen2cc']:
            # Check if path exits
            if not os.path.exists(FOREST):
                print("The forest path {} does not exist\n".format(FOREST))
                return
            log_parameters(log_file, forest_type, DEM, FOREST, radius, prob, winddir, windtol, pra_thd, sf)
        if forest_type in ['no_forest']:
            log_parameters(log_file, forest_type, DEM, DEM, radius, prob, winddir, windtol, pra_thd, sf)

        #######################
        # Calculate slope, vector roughness, and  windshelter
        #######################

        dem_data, transform, crs = read_dem_raster(DEM)

        print("Calculating slope angle")
        slope_deg = calculate_slope(dem_data, transform)
        slopeC = cauchy_slope_function(slope_deg)

        print("Calculating windshelter")
        windshelter = calculate_windshelter(dem_data, transform, radius, prob, winddir, windtol)
        windshelterC = cauchy_windshelter_function(windshelter)

        print("Vector Roggednes")
        aspect_deg=
        vector_roughness = calculate_ruggedness(dem_data, transform, 10, 5)
        vector_roughnessC = cauchy_ruggedness_function(vector_roughness)

        print("Calculating forest")
        if forest_type in ['stems']:
            forest_data, _, _ = read_dem_raster(FOREST)
        elif forest_type in ['bav', 'sen2cc']:
            # Handle the specific cases of 'bav' and 'sen2cc'
            forest_data, _, _ = read_dem_raster(FOREST)
        else:
            # Handle the case of 'pcc' and 'no_forest'
            forest_data, _, _ = read_dem_raster(DEM)
        forestC = cauchy_forest_function(forest_data, forest_type)

        #######################
        # --- Fuzzy logic operator
        #######################

        print("Starting the Fuzzy Logic Operator")
        PRA = fuzzy_logic_operator(slopeC, windshelterC, forestC)
        # Save raster to path using meta data from DEM.tif (i.e. projection)
        save_raster_as_geotiff(PRA, transform, crs, 'PRA/PRA_continous.tif')
        # Reclassify PRA to be used as input for FlowPy
        PRA = reclassify_PRA(PRA, pra_thd)
        # Save raster to path using meta data from DEM.tif (i.e. projection)
        save_raster_as_geotiff(PRA, transform, crs, 'PRA/PRA_binary.tif')
        # Remove islands smaller than 3 pixels
        remove_islands(sf)
        print('PRA complete')
        log_stop_time(log_file)

    except Exception as e:
        print("Error:", str(e))

    finally:
        log_stop_time(log_file)
        log_file.close()

PRA('pcc', 'DEM.tif', 'fores_density.tif',6, 0.5, 0, 180, 0.15, 3)

# if __name__ == "__main__":
#     forest_type = str(sys.argv[1])
#     if forest_type in ['pcc', 'stems', 'bav']:
#         DEM = sys.argv[2]
#         FOREST = sys.argv[3]
#         radius = int(sys.argv[4])
#         prob = float(sys.argv[5])
#         winddir = int(sys.argv[6])
#         windtol = int(sys.argv[7])
#         pra_thd = float(sys.argv[8])
#         sf = int(sys.argv[9])
#         PRA(forest_type, DEM, FOREST, radius, prob, winddir, windtol, pra_thd, sf)
#     if forest_type in ['no_forest']:
#         DEM = sys.argv[2]
#         radius = int(sys.argv[3])
#         prob = float(sys.argv[4])
#         winddir = int(sys.argv[5])
#         windtol = int(sys.argv[6])
#         pra_thd = float(sys.argv[7])
#         sf = int(sys.argv[8])
#         PRA(forest_type, DEM, DEM, radius, prob, winddir, windtol, pra_thd, sf)