import fnmatch
import numpy as np
import os
import sys
from imgReading import read_all_images, read_image
from PCA import PCAFaceRecognizer


def combination(distances, labels, distances3D, labels3D):
    min_dist = np.finfo('float').max
    label = -1
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] == labels3D[j]:
                ind = j
                break
        dist = distances[i]*distances3D[ind]
        if min_dist > dist:
            label = labels[i]
            min_dist = dist
    return label, min_dist


if __name__ == '__main__':
    work_dir = os.path.abspath(os.path.join('Databases', 'Data'))
    if not os.path.isdir(work_dir):
        print('No Data directory.')
        sys.exit(1)
    os.chdir(work_dir)
    # Create a train array with 2D photos
    XTrain, labelsTrain, _ = read_all_images(os.path.join(work_dir, 'train'))
    # 8 seems the best (6 for combination)
    recognizer = PCAFaceRecognizer(XTrain, labelsTrain, 8)
    # Create a train array with depth maps
    XTrain3D, labelsTrain3D, _ = read_all_images(os.path.join('train', '3D'))
    recognizer3D = PCAFaceRecognizer(XTrain3D, labelsTrain3D, 4)

    # Test recognzers
    recognized = 0.0
    recognized3D = 0.0
    recognized23D = 0.0
    directory = os.path.join(work_dir, 'test')
    directory3D = os.path.join(directory, '3D')
    filenames = fnmatch.filter(os.listdir(directory), '*.png')
    correct = '{0}: The person {1} is correctly recognized with distance {2:.2f}.\n'
    wrong = '{0}: The person {1} is wrongly recognized with guess {2}.\n'
    for i, filename in enumerate(filenames):
        filename3D = filename.rstrip(".png") + "_map.png"

        face, label_actual = read_image(os.path.join(directory, filename))
        face3D, label_actual3D = read_image(os.path.join(directory3D, filename3D))
        # error if wrong file has been found
        if label_actual != label_actual3D:
            print("ERROR with labels, filename {0}".format(filename))

        # predictions
        label_predicted, conf, distances = recognizer.predict(face)
        label_predicted3D, conf3D, distances3D = recognizer3D.predict(face3D)
        # simple judgement
        if label_actual == label_predicted:
            recognized = recognized + 1
            print(correct.format('2D', label_actual, conf))
        else:
            print(wrong.format('2D', label_actual, label_predicted))
        if label_actual3D == label_predicted3D:
            recognized3D = recognized3D + 1
            print(correct.format('3D', label_actual, conf3D))
        else:
            print(wrong.format('3D', label_actual, label_predicted3D))

        # combined approach
        label_predicted23D, conf23D = combination(distances, labelsTrain, distances3D, labelsTrain3D)
        if label_actual == label_predicted23D:
            recognized23D = recognized23D + 1
            print(correct.format('2D+3D', label_actual, conf23D))
        else:
            print(wrong.format('2D+3D', label_actual, label_predicted23D))

    # efficiency
    conf2D = recognized/len(filenames)*100.
    conf3D = recognized3D/len(filenames)*100.
    conf23D = recognized23D/len(filenames)*100.
    str_conf = "{0} recognized the right person for {1:.2f}%."
    print(str_conf.format('2D', conf2D))
    print(str_conf.format('3D', conf3D))
    print(str_conf.format('2D+3D', conf23D))
