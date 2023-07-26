import numpy as np
import rasterio
from osgeo import gdal
import os
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import convolve
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
    log_file_path = os.path.join(os.getcwd(), "data", "log.txt")
    
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

def log_stop_time(log_file):
    print('PRA complete')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    log_file.write("Stop time = {}\n".format(current_time))
    log_file.close()

def log_parameters(log_file, forest_type, DEM, FOREST, SLOPE, ASPECT, radius, prob, winddir, windtol, pra_thd, sf):
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
    log_file.write("forest_type: {}, DEM: {}, FOREST: {}, SLOPE: {}, ASPECT: {}, radius: {}, prob: {}, winddir: {}, windtol {}, pra_thd: {}, sf: {}\n".format(
        forest_type, DEM, FOREST, SLOPE, ASPECT, radius, prob, winddir, windtol, pra_thd, sf))

def create_path_if_not_exists(path):
    """
    Create the given path if it does not exist.

    Parameters:
        path (str): The path to be created if it does not exist.
        
    """
    # Create the directory at the specified path if it does not already exist
    # The 'exist_ok=True' argument ensures that the function does not raise an error if the directory already exists.
    os.makedirs(path, exist_ok=True)

def read_raster(RASTER):
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
    with rasterio.open(RASTER) as src:
        # Read the elevation values from band 1
        data = src.read(1)
        data = data.astype('float')
        
        # Replace invalid values (e.g., values less than -100) with nodata value (-9999)
        data[np.where(data < -100)] = -9999
        

        # Get the profile of the raster dataset
        profile = src.profile
        profile.update({"dtype": "float32", "nodata": -9999})
 
    # Retrieve the transformation and CRS from the profile
    #transform = profile['transform']
    #crs = profile['crs']

    return data, profile

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

def calculate_windshelter(dem_data, profile, radius, prob, winddir, windtol):
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

    def windshelter_prep(transform, radius, direction, tolerance):
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
        dist = (np.sqrt((x_arr - cell_center[0]) ** 2 + (y_arr - cell_center[1]) ** 2)) * transform[0]

        # Calculate the boolean mask representing the circular sector using the sector_mask function
        mask = sector_mask(dist.shape, (radius, radius), radius, (direction-tolerance+270, direction+tolerance+270))

        # Correct a bug in the mask where the center cell is not included in the circular sector
        mask[radius, radius] = True

        return dist, mask

    def windshelter(x, profile, prob, dist, mask, radius):
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
        data[data==profile['nodata']]=np.nan
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

    def windshelter_window(dem_data, profile, radius, prob):
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
        dist, mask = windshelter_prep(profile['transform'], radius, winddir, windtol)

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
                data = windshelter(data, profile , prob, dist, mask, radius).tolist()

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
    data = windshelter_window(dem_data, profile, radius, prob)

    # Replace any NaN values in the windshelter data with -9999
    data = np.nan_to_num(data, nan=-9999)

    # Return the windshelter data
    return data

def calculate_ruggedness(slope_data, aspect_data, window=3):
    """
    Calculate vector ruggedness raster of a Digital Elevation Model (DEM) using the Sappington method.

    Parameters:
        slope_raster (numpy.ndarray): 2D array representing the slope raster in degrees.
        aspect_raster (numpy.ndarray): 2D array representing the aspect raster in degrees.

    Returns:
        ruggedness: The vector ruggedness raster.
    """
    
    # Convert to radians
    slope_rad = np.radians(slope_data)
    aspect_rad = np.radians(aspect_data)

    # Calculate xyz components
    xy_raster = np.sin(slope_rad)
    z_raster = np.cos(slope_rad)
    x_raster = np.sin(aspect_rad) * xy_raster
    y_raster = np.cos(aspect_rad) * xy_raster

    # Define the 3x3 focal neighborhood kernel for sum
    kernel = np.ones((window, window))

    # Calculate the x, y, and z sum rasters using focal sum
    xsum_raster = convolve(x_raster, kernel, mode='constant', cval=0.0)
    ysum_raster = convolve(y_raster, kernel, mode='constant', cval=0.0)
    zsum_raster = convolve(z_raster, kernel, mode='constant', cval=0.0)

    # Calculate the vector ruggedness raster
    ruggedness = 1 - np.sqrt(xsum_raster**2 + ysum_raster**2 + zsum_raster**2) / 9

    return ruggedness

def cauchy_slope_function(slope_deg):
    """Calculate the Cauchy function for slope."""  
    a = 11
    b = 4
    c = 43
    #f.write("Cauchy slope function: a={}, b={}, c={}\n".format(a, b, c))
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

def cauchy_ruggedness_function(ruggedness):
    """Calculate the Cauchy function for terrain_ruggedness."""
    # --- Define bell curve parameters for vector roughness
    a = 0.01
    b = 5
    c = -0.007
    #f.write("Cauchy vector roughness function: a={}, b={}, c={}\n".format(a, b, c))
    # Calculate the Cauchy function for vector roughness
    ruggednessC = 1 / (1 + ((ruggedness - c) / a) ** (2 * b))
    ruggednessC = np.round(ruggednessC, 5)
    return ruggednessC

def cauchy_forest_function(forest, forest_type):
    """Calculate the Cauchy function for the given forest type."""
    if forest_type in ['stems', 'bav', 'sen2cc', ['pcc']]:
        if forest_type in ['stems']:
            a = 350
            b = 2
            c = -120
            #f.write("Cauchy forest function (stems): a={}, b={}, c={}\n".format(a, b, c))
        elif forest_type in ['bav']:
            a = 20
            b = 3.5
            c = -10
            #f.write("Cauchy forest function (bav): a={}, b={}, c={}\n".format(a, b, c))
        elif forest_type in ['sen2cc']:
            a = 50 # still finalizing defualts for Sen2cc, likeily will be region dependent based on local forest structure
            b = 1.5
            c = 0
            #f.write("Cauchy forest function (sen2cc): a={}, b={}, c={}\n".format(a, b, c)) 
        # --- Define bell curve parameters for percent canopy cover
        elif forest_type in ['pcc']:
            a = 40
            b = 3.5
            c = -15
            #if forest_type in ['pcc']:
                #f.write("Cauchy forest function (pcc): a={}, b={}, c={}\n".format(a, b, c))
        forestC = 1/(1+((forest-c)/a)**(2*b))
        # --- Ares with no forest and assigned -9999 will get a really small value which suggest dense forest. This function fixes this, but might have to be adjusted depending on the input dataset.
        forestC[np.where(forestC <= 0.00001)] = 1
        forestC = np.round(forestC, 5)
    else:
        forestC = np.where(forest >= -100, 1, forest)
    
    return forestC

def fuzzy_logic_operator(slopeC, windshelterC, ruggednessC, forestC):
    """Perform the fuzzy logic operator."""
    print("Starting the Fuzzy Logic Operator")

    minvar = np.minimum(slopeC, windshelterC)
    minvar = np.minimum(minvar, ruggednessC)
    minvar = np.minimum(minvar, forestC)

    PRA = (1-minvar)*minvar+minvar*(slopeC+windshelterC+forestC+ruggednessC)/4
    PRA = np.round(PRA, 5)
    PRA = PRA * 100
    return PRA

def reclassify_PRA(PRA, profile, pra_thd):
    """
    Reclassify PRA to be used as input for FlowPy.

    Parameters:
        PRA (numpy.ndarray): The 2D array containing the PRA data.
        pra_thd (float): The threshold value for reclassification.

    Returns:
        numpy.ndarray: The reclassified PRA data.
    """

    # Calculate the threshold value for reclassification
    pra_thd = pra_thd * 100

    # Reclassify PRA based on the threshold value
    PRA[np.where((0 <= PRA) & (PRA < pra_thd))] = 0
    PRA[np.where((pra_thd < PRA) & (PRA <= 100))] = 1

    return PRA

def remove_islands(sf): 
    sievefilter = sf + 1
    Image = gdal.Open('PRA_binary.tif', 1)  # open image in read-write mode
    Band = Image.GetRasterBand(1)
    gdal.SieveFilter(srcBand=Band, maskBand=None, dstBand=Band, threshold=sievefilter, connectedness=8, callback=gdal.TermProgress_nocb)
    del Image, Band  # close the datasets.

def PRA(forest_type, DEM, FOREST, SLOPE, ASPECT, radius, prob, winddir, windtol, pra_thd, sf):
    # --- Create log file
    log_file = create_log_file()
    try:
        ##########################
        # --- Check input files
        ##########################
        path = os.path.join(os.getcwd(), "data")
        create_path_if_not_exists(path)
        log_start_time(log_file)
        # Check if path exits
        if not os.path.exists(DEM):
            print("The DEM path {} does not exist".format(DEM))
            return
        if not os.path.exists(SLOPE):
            print("The SLOPE path {} does not exist".format(SLOPE))
            return
        if not os.path.exists(ASPECT):
            print("The ASPECT path {} does not exist".format(ASPECT))
            return
        if forest_type in ['pcc', 'stems', 'bav', 'sen2cc']:
            # Check if path exits
            if not os.path.exists(FOREST):
                print("The forest path {} does not exist\n".format(FOREST))
                return
            log_parameters(log_file, forest_type, DEM, FOREST, SLOPE, ASPECT, radius, prob, winddir, windtol, pra_thd, sf)
        if forest_type in ['no_forest']:
            log_parameters(log_file, forest_type, DEM, DEM, SLOPE, ASPECT, radius, prob, winddir, windtol, pra_thd, sf)

        #######################
        # Calculate slope, ruggedness, and  windshelter
        #######################

        dem_data, dem_profile = read_raster(DEM)
        slope_data, _ = read_raster(SLOPE)
        aspect_data, _ = read_raster(ASPECT)
        #ruggedness, _ = read_raster(RUGGEDNESS)
        if forest_type in ['pcc', 'bav', 'stems', 'sen2cc']:
            # Handle the case of 'pcc', 'bav', 'stems', 'sen2cc'
            forest_data, _ = read_raster(FOREST)
        else:
            forest_data, _ = read_raster(DEM)

        print("Calculating slope angle")
        # slope_deg = calculate_slope(dem_data, profile['transform']
        slopeC = cauchy_slope_function(slope_data)
        with rasterio.open('slopeC.tif', "w", **dem_profile) as dest:
             dest.write(slopeC, 1)

        print("Calculating windshelter")
        windshelter = calculate_windshelter(dem_data, dem_profile, radius, prob, winddir, windtol)
        with rasterio.open('wind_shelter.tif', "w", **dem_profile) as dest:
            dest.write(windshelter)
        windshelterC = cauchy_windshelter_function(windshelter)
        with rasterio.open('windshelterC.tif', "w", **dem_profile) as dest:
             dest.write(windshelterC)

        print("Calculating Ruggedness")
        ruggedness = calculate_ruggedness(slope_data, aspect_data)
        with rasterio.open('ruggedness.tif', "w", **dem_profile) as dest:
             dest.write(ruggedness, 1)
        ruggednessC = cauchy_ruggedness_function(ruggedness)
        with rasterio.open('ruggednessC.tif', "w", **dem_profile) as dest:
             dest.write(ruggednessC, 1)

        print("Calculating forest")
        forestC = cauchy_forest_function(forest_data, forest_type)
        with rasterio.open('forestC.tif', "w", **dem_profile) as dest:
             dest.write(forestC, 1)

        #######################
        # --- Fuzzy logic operator
        #######################

        print("Starting the Fuzzy Logic Operator")
        PRA = fuzzy_logic_operator(slopeC, windshelterC, slopeC, forestC)

        # Save raster to path using meta data from dem.tif (i.e. projection)
        with rasterio.open('PRA_continuous.tif', "w", **dem_profile) as dest:
            dest.write(PRA)
        
        PRA = reclassify_PRA(PRA, dem_profile, pra_thd)
        
        # Save raster to path using meta data from dem.tif (i.e. projection)
        with rasterio.open('PRA_binary.tif', "w", **dem_profile) as dest:
            dest.write(PRA)
        
        # Remove islands smaller than 3 pixels
        remove_islands(sf)
        
        print('PRA complete')
        log_stop_time(log_file)

    except Exception as e:
        print("Error:", str(e))

    finally:
        log_stop_time(log_file)

PRA('sen2cc', 'dem.tif', 'forest.tif', 'slope.tif', 'aspect.tif', 6, 0.5, 0, 180, 0.15, 3)

# if __name__ == "__main__":
#     forest_type = str(sys.argv[1])
#     if forest_type in ['pcc', 'stems', 'bav', 'sen2cc']:
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