
import numpy as np
from PIL import Image

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    :return: RGBImage of given file
    :param path: filepath of image
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Save the given RGBImage instance to the given path
    :param path: filepath of image
    :param image: RGBImage object to save
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    TODO: add description
    """

    def __init__(self, pixels):
        """
        TODO: add description

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        # YOUR CODE GOES HERE #
        # Raise exceptions here

        if type(pixels) != list or len(pixels) < 1:
            raise TypeError
        for i in pixels:
            if type(i) != list:
                raise TypeError
            if len(i) != len(pixels[0]):
                raise TypeError
            for j in i:
                if type(j) != list:
                    raise TypeError()
                if len(j) != 3:
                    raise TypeError()
                for k in j:
                    if k < 0 or k > 255:
                        raise ValueError

        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
      

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        # YOUR CODE GOES HERE #
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
    

        return [[list(pixel) for pixel in row] for row in self.pixels]

    def copy(self):
        """
        TODO: add description

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """

        return RGBImage.get_pixels(self)

    def get_pixel(self, row, col):
        """
        TODO: add description

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """

        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError()
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            raise ValueError()

        return tuple(self.pixels[row][col])

    def set_pixel(self, row, col, new_color):
        """
        TODO: add description

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        # YOUR CODE GOES HERE #

        if isinstance(row, int) == False or isinstance(col, int) == False:
            raise TypeError()
        if row >= self.num_rows or col >= self.num_cols:
            raise ValueError()
        if type(new_color) != tuple:
            raise TypeError()
        if len(new_color) != 3:
            raise TypeError()
        if not all(isinstance(i, int) for i in new_color):
            raise TypeError()

        for i in range(len(new_color)):
            if new_color[i] > 255:
                raise ValueError()
            elif new_color[i] >= 0:
                self.pixels[row][col][i] = new_color[i]
            else:
                continue



# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    TODO: add description
    """

    def __init__(self):
        """
        TODO: add description

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0

    def get_cost(self):
        """
        TODO: add description

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        return self.cost

    def negate(self, image):
        """
        TODO: add description

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img_input = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img_input)
        >>> id(img_input) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output,
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img_input = img_read_helper('img/gradient_16x16.png')           # 2
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')  # 3
        >>> img_negate = img_proc.negate(img_input)                         # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/gradient_16x16_negate.png', img_negate)# 6
        """
        # YOUR CODE GOES HERE #

        negate_pixels = [[[255 - c for c in pixel] for pixel in row]\
        for row in image.get_pixels()]
        negate_image = RGBImage(negate_pixels)
        return negate_image

    def grayscale(self, image):
        """
        TODO: add description

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_gray.png')
        >>> img_gray = img_proc.grayscale(img_input)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/gradient_16x16_gray.png', img_gray)
        """
        # YOUR CODE GOES HERE #

        gray_pixels = [[[sum(pixel)//3 for c in pixel] for pixel in row] for row in image.get_pixels()]
        gray_image = RGBImage(gray_pixels)
        return gray_image

    def rotate_180(self, image):
        """
        TODO: add description

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img_input)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/gradient_16x16_rotate.png', img_rotate)
        """
        # YOUR CODE GOES HERE #
        reversed_pixels = [list(row)[::-1] for row in image.get_pixels()][::-1]
        # print(reversed_pixels)
        return RGBImage(reversed_pixels)


# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    TODO: add description
    """

    def __init__(self):
        """
        TODO: add description

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0
        self.amount = 0
        self.rotate180_count = 0

    def negate(self, image):
        """
        TODO: add description

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')
        >>> img_negate = img_proc.negate(img_input)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        # YOUR CODE GOES HERE #

        if self.amount > 0:
            self.amount -= 1
        else:
            self.cost += 5
        return super().negate(image)

    def grayscale(self, image):
        """
        TODO: add description

        """
        # YOUR CODE GOES HERE #
        if self.amount > 0:
            self.amount -= 1
        else:
            self.cost += 6
        return super().grayscale(image)

    def rotate_180(self, image):
        """
        TODO: add description

        # Check that the cost is 0 after two rotation calls
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        10
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        # YOUR CODE GOES HERE #

        if self.amount > 0:
            self.amount -= 1
        else:
            if self.rotate180_count%2 == 0:
                self.cost += 10
            if self.rotate180_count%2 == 1:
                self.cost -= 10
        self.rotate180_count += 1
        return super().rotate_180(image)

    def redeem_coupon(self, amount):
        """
        TODO: add description

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        0
        """

        # YOUR CODE GOES HERE #

        if amount <= 0:
            raise ValueError
        if type(amount) != int:
            raise TypeError
        self.amount += amount



# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    TODO: add description
    """

    def __init__(self):
        """
        TODO: add description

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        TODO: add description

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> img_in_back = img_read_helper('img/gradient_16x16.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_16x16_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_16x16_chroma.png', img_chroma)
        """
        # YOUR CODE GOES HERE #
        # chroma = RGBImage(chroma_image).get_pixels()
        # background = RGBImage(background_image).get_pixels()

        if type(chroma_image) != RGBImage or type(background_image) != RGBImage:
            raise TypeError
        if chroma_image.size() != background_image.size():
            raise ValueError

        new_chroma_image = RGBImage(chroma_image.copy())
        for i in range (len(new_chroma_image.pixels)):
            for j in range(len(new_chroma_image.pixels[i])):
                if new_chroma_image.get_pixel(i, j) == color:
                    new_chroma_image.set_pixel(i, j, background_image.get_pixel(i, j))
        return new_chroma_image

    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        TODO: add description

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (15, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/square_16x16_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/square_16x16_sticker.png', img_combined)
        """
        # YOUR CODE GOES HERE #

        if type(sticker_image) != RGBImage or type(background_image) != RGBImage:
            raise TypeError
        if sticker_image.size()[0] >= background_image.size()[0] or sticker_image.size()[1] >= background_image.size()[1]:
            raise ValueError
        if type(sticker_image.size()[0]) != int or type(sticker_image.size()[1]) != int:
            raise TypeError
        if x_pos + sticker_image.size()[0] > background_image.size()[0] or y_pos + sticker_image.size()[1] > background_image.size()[1]:
            raise ValueError

        # new_background_image = background_image.copy()
        new_background_image = RGBImage(background_image.copy())
        size_sticker = sticker_image.size()
        for i in range(x_pos, len(new_background_image.pixels)):
            for j in range(y_pos, len(new_background_image.pixels)):
                if i in range(x_pos, size_sticker[0] + x_pos) and j in range(y_pos, size_sticker[1] + y_pos):
                    new_background_image.set_pixel(i, j, sticker_image.get_pixel(i - x_pos, j - y_pos))
        return new_background_image


# Part 5: Image KNN Classifier #

def create_random_pixels(low, high, nrows, ncols):
        """
        Create a random pixels matrix with dimensions of
        3 (channels) x `nrows` x `ncols`, and fill in integer
        values between `low` and `high` (both exclusive).
        """
        return np.random.randint(low, high + 1, (nrows, ncols, 3)).tolist()

class ImageKNNClassifier:
    """
    TODO: add description

    # make random training data (type: List[Tuple[RGBImage, str]])
    >>> train = []

    # create training images with low intensity values
    >>> train.extend(
    ...     (RGBImage(create_random_pixels(0, 75, 300, 300)), "low")
    ...     for _ in range(20)
    ... )

    # create training images with high intensity values
    >>> train.extend(
    ...     (RGBImage(create_random_pixels(180, 255, 300, 300)), "high")
    ...     for _ in range(20)
    ... )

    # initialize and fit the classifier
    >>> knn = ImageKNNClassifier(5)
    >>> knn.fit(train)

    # should be "low"
    >>> print(knn.predict(RGBImage(create_random_pixels(0, 75, 300, 300))))
    low

    # can be either "low" or "high" randomly
    >>> print(knn.predict(RGBImage(create_random_pixels(75, 180, 300, 300))))
    This will randomly be either low or high

    # should be "high"
    >>> print(knn.predict(RGBImage(create_random_pixels(180, 255, 300, 300))))
    high
    """

    def __init__(self, n_neighbors):
        """
        TODO: add description
        """
        # YOUR CODE GOES HERE #
        self.n_neighbors = n_neighbors
        self.data = []
        self.call_fit = 0

    def fit(self, data):
        """
        TODO: add description
        """
        # YOUR CODE GOES HERE #

        if len(data) <= self.n_neighbors:
            raise ValueError
        if len(self.data) > 0:
            raise ValueError

        self.data = data
        self.call_fit += 1

    @staticmethod
    def distance(image1, image2):
        """
        TODO: add description
        """
        # YOUR CODE GOES HERE #

        if isinstance(image1, RGBImage) == False or isinstance(image2, RGBImage) == False:
            raise TypeError
        if image1.size() != image2.size():
            raise ValueError

        pixel_distance = sum([(image1.pixels[i][j][k] - image2.pixels[i][j][k])**2 \
    for i in range(len(image1.pixels)) for j in range(len(image1.pixels[i])) for k in range(len(image1.pixels[i][j]))])
        row_distance = sum([(len(image1.pixels[i]) - len(image2.pixels[i]))**2 for i in range(len(image1.pixels))])
        col_distance = sum([(len(image1.pixels[i][j]) - len(image2.pixels[i][j]))**2 for i in range(len(image1.pixels)) for j in range(i)])
        distance = pixel_distance + row_distance + col_distance
        distance = distance**(1/2)

        return distance


    @staticmethod
    def vote(candidates):
        """
        TODO: add description
        """
        # YOUR CODE GOES HERE #

        counter = 0
        name = ""

        for i in candidates:
            count  = candidates.count(i)
        if count > counter:
            counter = count
            name = i

        return name


    def predict(self, image):
        """
        TODO: add description
        """
        # YOUR CODE GOES HERE #

        if self.call_fit == 0:
            raise ValueError

        data = [list(i) for i in self.data]
        # for i in self.data:
        #     data.append(list(i))
        
        data = list(map(lambda x: [ImageKNNClassifier.distance(image, x[0]), x[1]], data))
        # for i in data:
        #     i[0] = ImageKNNClassifier.distance(image, i[0])


        data = sorted(data, key = lambda data: data[0])
        data_list = data[:self.n_neighbors][1]

        return ImageKNNClassifier.vote(data_list)
