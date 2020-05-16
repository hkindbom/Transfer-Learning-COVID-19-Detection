""" 
DD2424 Deep Learning in Data Science
May 2020
This script was written to handle the data used in the group project
"""
class Dataset:
    def __init__(
        self,
        test_img_dir,
        train_img_dir,
        test_img_descriptions_file,
        train_img_descriptions_file,
        input_shape = (224, 224),
        batch_size = 10
                ):

        self.test_img_dir = test_img_dir
        self.train_img_dir = train_img_dir
        self.test_img_descriptions = get_file_lines(test_img_descriptions_file)
        self.train_img_descriptions = get_file_lines(train_img_descriptions_file)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.batch_nr = 1
        self.max_batch = len(self.train_img_descriptions) // self.batch_size
        self.mapping = {
                'normal': 0,
                'pneumonia': 1,
                'COVID-19': 2
                }

        self.y_train = None
        self.y_test = None
        self.x_batch = None
        self.y_batch = None

    
    def get_current_batch(self):
        return self.x_batch, self.y_batch
    
    def _get_class_dist(self, y):
        class_dist = {}
        for class_name in self.mapping:
            class_dist[class_name] = np.count_nonzero(y == self.mapping[class_name])
        return class_dist

    def get_test_class_dist(self):
        test_class_dist = self._get_class_dist(self.y_test)

        return test_class_dist

    def read_test_data(self):
        self.x_test, self.y_test = self._read_correct_images(self.test_img_dir, self.test_img_descriptions)

    def _get_label(self, img_desc_list):
        for class_name in self.mapping:
            if class_name in img_desc_list:
                return self.mapping[class_name]
            
    # Must load training data in batches since memory error otherwise    
    def next_train_batch(self):
        
        if self.batch_nr == self.max_batch:
            print('No data left')
            return [], []
        
        start_img = (self.batch_nr-1) * self.batch_size
        end_img = (self.batch_nr) * self.batch_size        
                
        batch_descriptions = self.train_img_descriptions[start_img: end_img]
        
        self.x_batch, self.y_batch = self._read_correct_images(self.train_img_dir, batch_descriptions)
        
        self.batch_nr += 1
        

    def _read_correct_images(self, img_dir, img_descriptions):
        data_path = os.path.join(img_dir, '*g')
        images = glob.glob(data_path)
        
        x_list = []
        y_list = []
        
        for image_path in images:
            image_name = image_path.replace(img_dir + '/', '')
            for img_desc in img_descriptions:
                if image_name in img_desc:
                    img_desc_list = img_desc.split()
                    label = self._get_label(img_desc_list)
                    y_list.append(label)

                    img_array = cv2.imread(image_path)
                    resized_img_array = cv2.resize(img_array, self.input_shape)
                    x_list.append(resized_img_array)
                    
        y_array = np.array(y_list)
        x_array = np.array(x_list)        

        return x_array, y_array