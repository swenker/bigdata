__author__ = 'wenjusun'


"""
http://stats.stackexchange.com/questions/82923/mixing-continuous-and-binary-data-with-linear-svm?newreg=5d0ce9b912074693adfb8ffc9b3260b1
"""
class DataDict:
    workclass = {'Private'.lower(): 0, 'Self-emp-not-inc'.lower(): 1, 'Self-emp-inc'.lower(): 2,
                 'Federal-gov'.lower(): 3, 'Local-gov'.lower(): 4, 'State-gov'.lower(): 5,
                 'Without-pay'.lower(): 6, 'Never-worked'.lower(): 7
    }

    """
    an ordered factor with levels Preschool < 1st-4th < 5th-6th < 7th-8th < 9th < 10th < 11th < 12th < HS-grad < Prof-school < Assoc-acdm < Assoc-voc < Some-college < Bachelors < Masters < Doctorate.
    """
    education = {'Doctorate'.lower(): 15, 'Masters'.lower(): 14, 'Bachelors'.lower(): 13, 'Some-college'.lower(): 12,
                 'Assoc-voc'.lower(): 11, 'Assoc-acdm'.lower(): 10, 'Prof-school'.lower(): 9, 'HS-grad'.lower(): 8,
                 '12th'.lower(): 7, '11th'.lower(): 6, '10th'.lower(): 5, '9th'.lower(): 4, '7th-8th'.lower(): 3,
                 '5th-6th'.lower(): 2, '1st-4th'.lower(): 1, 'Preschool'.lower(): 0
    }

    "marital-status"
    marital_status = {'Married-civ-spouse'.lower(): 0, 'Divorced'.lower(): 1, 'Never-married'.lower(): 2,
                      'Separated'.lower(): 3, 'Widowed'.lower(): 4,
                      'Married-spouse-absent'.lower(): 5, 'Married-AF-spouse'.lower(): 6
    }
    occupation = {'Tech-support'.lower(): 0, 'Craft-repair'.lower(): 1, 'Other-service'.lower(): 2,
                  'Sales'.lower():3, 'Exec-managerial'.lower(): 4,
                  'Prof-specialty'.lower(): 5, 'Handlers-cleaners'.lower(): 6, 'Machine-op-inspct'.lower(): 7,
                  'Adm-clerical'.lower(): 8,
                  'Farming-fishing'.lower(): 9, 'Transport-moving'.lower(): 10, 'Priv-house-serv'.lower(): 11,
                  'Protective-serv'.lower(): 12,
                  'Armed-Forces'.lower(): 13}

    relationship = {'Wife'.lower(): 0, 'Own-child'.lower(): 1, 'Husband'.lower(): 2, 'Not-in-family'.lower(): 3,
                    'Other-relative'.lower(): 4, 'Unmarried'.lower(): 5}

    race = {'White'.lower(): 0, 'Asian-Pac-Islander'.lower(): 1, 'Amer-Indian-Eskimo'.lower(): 2, 'Other'.lower(): 3,
            'Black'.lower(): 4}

    sex = {'Female'.lower(): 0, 'Male'.lower(): 1}

    # 41 countries and district
    native_country = {'United-States'.lower(): 0, 'Cambodia'.lower(): 1, 'England'.lower(): 2, 'Puerto-Rico'.lower(): 3,
                      'Canada'.lower(): 4, 'Germany'.lower(): 5, 'Outlying-US(Guam-USVI-etc)'.lower(): 6,
                      'India'.lower(): 7, 'Japan'.lower(): 8, 'Greece'.lower(): 9, 'South'.lower(): 10,
                      'China'.lower(): 11, 'Cuba'.lower(): 12, 'Iran'.lower(): 13, 'Honduras'.lower(): 14,
                      'Philippines'.lower(): 15, 'Italy'.lower(): 16, 'Poland'.lower(): 17,
                      'Jamaica'.lower(): 18, 'Vietnam'.lower(): 19, 'Mexico'.lower(): 20, 'Portugal'.lower(): 21,
                      'Ireland'.lower(): 22, 'France'.lower(): 23, 'Dominican-Republic'.lower(): 24, 'Laos'.lower(): 25,
                      'Ecuador'.lower(): 26, 'Taiwan'.lower(): 27, 'Haiti'.lower(): 28, 'Columbia'.lower(): 29,
                      'Hungary'.lower(): 30, 'Guatemala'.lower(): 31, 'Nicaragua'.lower(): 32, 'Scotland'.lower(): 33,
                      'Thailand'.lower(): 34, 'Yugoslavia'.lower(): 35, 'El-Salvador'.lower(): 36,
                      'Trinadad&Tobago'.lower(): 37, 'Peru'.lower(): 38,
                      'Hong'.lower(): 39, 'Holand-Netherlands'.lower(): 40}

    income_label = {'<=50K': 0, '>50K': 1}

    def get_feature_labels(self):
        feature_lables=[]
        feature_lables.append('age')
        feature_lables.extend(self.workclass.keys())
        feature_lables.extend(self.education.keys())
        feature_lables.extend(self.marital_status.keys())
        feature_lables.extend(self.occupation.keys())
        feature_lables.extend(self.relationship.keys())
        feature_lables.extend(self.race.keys())
        feature_lables.extend(self.sex.keys())
        feature_lables.append('capital_gain')
        feature_lables.append('capital_loss')
        feature_lables.append('hours_per_week')
        feature_lables.extend(self.native_country.keys())

        return feature_lables


    # TODO float
    @staticmethod
    def rescale_continuous_to_1(input_value, max, min,mean,sd):
        # return round((2.0 * input_value - max - min) / (max - min),2)
        #TODO x = (x-u)/theta  u=mean value  theta=standard deviation
        return round( (1.0 * input_value - min) / (max - min),1)

        # rescaled_value = (2.0 * input_value - max - min) / (max - min)
        # normalized_value = round((rescaled_value-mean)/sd,2)
        # return normalized_value
        #error rate is more higher,why?

    @staticmethod
    def reverse_rescaled_value(new_value,max,min):
        return (max-min)*new_value+min

    def get_workclass_vector(self, workclass):
        workclass_vector = [0] * len(self.workclass)
        if workclass:
            workclass_vector[self.workclass.get(workclass.lower())] = 1

        return workclass_vector

    def get_education_vector(self, edu):
        edu_vector = [0] * len(self.education)
        if edu:
            edu_vector[self.education.get(edu.lower())] = 1
        return edu_vector

    def get_marital_status_vector(self, marital_status):
        marital_status_vector = [0] * len(self.marital_status)
        if marital_status:
            marital_status_vector[self.marital_status.get(marital_status.lower())] = 1
        return marital_status_vector

    def get_occupation_vector(self, occupation):
        occupation_vector = [0] * len(self.occupation)
        if occupation:
            # print occupation
            occupation_vector[self.occupation.get(occupation.lower())] = 1
        return occupation_vector

    def get_relationship_vector(self, relationship):
        relationship_vector = [0] * len(self.relationship)
        if relationship:
            relationship_vector[self.relationship.get(relationship.lower())] = 1
        return relationship_vector

    def get_race_vector(self, race):
        race_vector = [0] * len(self.race)
        if race:
            # print race
            race_vector[self.race.get(race.lower())] = 1
        return race_vector

    def get_sex_vector(self, sex):
        sex_vector = [0] * len(self.sex)
        if sex:
            sex_vector[self.sex.get(sex.lower())] = 1
        return sex_vector

    def get_native_country(self, country):
        native_country_vector = [0] * len(self.native_country)
        if (country):
            native_country_vector[self.native_country.get(country.lower())] = 1
        return native_country_vector

    def get_income_large_50k(self, label_str):
        result = 0
        if label_str:
            result = self.income_label.get(label_str)
        return result


    def get_all_labels(self,label_file = r'C:\ZZZZZ\0-sunwj\bigdata\data\adult-income\labels.txt'):
        labels=[]
        with open(label_file) as f:
            for line in f:
                labels.append(int(line.strip()))
        return labels

    # def load_test_set(self):

    def get_training_set(self):
        dataset=[
            [-0.4, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, -0.57, -1.0, -0.2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, -1.0, -1.0, -0.76, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.42, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, -1.0, -1.0, -0.2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.01, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, -1.0, -1.0, -0.2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, -1.0, -1.0, -0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.45, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1.0, -1.0, -0.2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.12, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, -1.0, -1.0, -0.69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.04, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, -1.0, -1.0, -0.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.62, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1.82, -1.0, 0.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.32, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0.04, -1.0, -0.2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.45, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, -1.0, -1.0, 0.61, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.64, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1.0, -1.0, -0.2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.84, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1.0, -1.0, -0.41, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.59, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, -1.0, -1.0, 0.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.37, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1.0, -1.0, -0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.53, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, -1.0, -1.0, -0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.78, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, -1.0, -1.0, -0.31, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.59, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, -1.0, -1.0, -0.2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.42, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, -1.0, -1.0, 0.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-0.29, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, -1.0, -1.0, -0.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1,]

        return dataset,labels

if __name__=='__main__':
    print DataDict.reverse_rescaled_value(0.29,90,17)
    print DataDict.reverse_rescaled_value(0.3,90,17)
    print DataDict.reverse_rescaled_value(0.44,90,17)
    print DataDict.reverse_rescaled_value(0.48,90,17)
    print DataDict.reverse_rescaled_value(0.34,90,17)
    print DataDict.reverse_rescaled_value(0.19,90,17)