# Multilabel Classifier for Injury using BERT
## Description 
The goal is to classify the Cause and Body Part from the Description of an Injury usin the BERT model.
Processed the dataset and saved them in a csv file. Then for training, bert-base-uncased model from HuggingFace was used.
Two models were build one for classifying Cause and another for classifying Body Part. 

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/26bf4923-12eb-4a62-8356-22eb2384587a)

## Directory Structure:
 
The Jupyter notebook files are in the Model Training folder. The train.csv and test.csv files are inside the directory ‘data’. The data directory is also used in storing the processed csv files that will be loaded later in model training

## Preprocessing the Dataset / Exploratory Data Analysis – EDA.ipynb:
Loading the train.csv file we can see that it has 6 columns [LossDescription, ResultingInjuryDesc, PartInjuredDesc, Cause - Hierarchy 1, Body Part - Hierarchy 1, Index], and loaded using the index column as index. 

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/340cf37a-b3c9-45cf-a157-1c2927b5a97b)


The texts in [LossDescription, ResultingInjuryDesc, PartInjuredDesc] were merged into a single column called ‘description’ and dropped.

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/343ad528-7027-459c-83bb-a6f3214afb26)

Now our dataframe only has 3 columns namely [Cause - Hierarchy 1, Body Part - Hierarchy 1, description].

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/87b8bfcc-caa6-48cf-96af-0c20d13708b2)


The text inside the dataframe is converted to lowercase.

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/baaf4c88-968d-483d-b75c-e38e7375c4ce)


Then we split the dataframe into two separate dataframes, because we need to classify two tables separately. 
Once we have split, the null values are dropped from the two dataframes.

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/03fa9d7f-c810-4935-825c-16fb7445266b)
![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/714cff31-916e-4b8d-a972-cf8285e59900)

We can see that there are 12 values in Cause - Hierarchy 1 and 7 values in Body Part - Hierarchy 1.

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/59722eb6-26b3-4daa-8f12-6637c1aad49d)

Now the text values are converted into labels in a new column called label.

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/09a21981-f9d7-495d-b39a-a6ddeac78cdc)

These dataframes are them saved as csv files in side the data directory only containing the columns [description, label].

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/82f6c6d9-6f5c-4912-9c73-851d67a09c32)

## Model Training – cause_train.ipynb, bodypart_train.ipynb:
I used the bert-base-uncased pre-trained model from HuggingFace for classification. I loaded the model and finetuned it in pytorch using the dataset so that it can classify the class labels.
Loaded the processed dataset, split the dataset into train and test set then tokenized the description column so that it will be accepted by the model.

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/02c9bc0f-88e9-45d4-a3f4-0ac6ef15b25f)

Removed the Unnecessary columns

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/a0b9cc90-553d-4285-a36d-723c292751d3)

 
Loaded the data as a pytroch dataloader object

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/e0c83117-7247-4952-a7b9-668d096ac23a)


Loaded the Bert model

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/21f7a618-7185-45b7-bf64-aaeec9ec1a81)

 
Initializing the optimizer and Learning rate scheduler

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/4ab8a80c-2a37-4a06-8d92-c5cbc0abac33)

Finetuning the Pretrained model

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/76dd3415-40d5-4b27-a2aa-df64439a0421)

This method was used for both the models namely the cause_model to classify Cause - Hierarchy 1 and bodypart_model to classify Body Part - Hierarchy 1.

## Evaluation:
I got the Following accuracy on the Validation set for the respective models
Model for Classifying Cause - Hierarchy 1

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/05194444-fe54-4748-bc02-672a71f07127)

 
Saving the model as cause_model

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/03f47ed1-c642-4245-a546-8f578a884a77)

 
Model for Classifying Body Part - Hierarchy 1

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/aeda4669-9366-4551-b773-589e89ed1c22)

 
Saving this model as bodypart_model

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/403ebffd-4680-4046-8b81-4abc7fa85655)

 

## Model Inference / Prediction – test.ipynb:

The Prediction follows the same method as before like processing the dataset and loading it.
Processing the test data and saving it in a csv file.

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/93a0c522-39f7-48c1-9ba3-62aebc382f61)

 
Loading the processed test data and tokenizing it.

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/23b43d5c-017e-471e-8717-4cc23e15a80a)

 
Loading both of the saved classification models

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/debc2e60-4506-42ce-b994-87da7f4dff92)

Running the inference and obtaining the prediction.
![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/71fe2ed1-311b-40cc-bdc0-adb213d7e72d)

 
Saving the predictions into the test.csv dataset.

![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/d83515f4-05e6-4ecb-b0c3-d1e363d5802b)
![image](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/e7dd00c2-4740-415f-8eb6-3b5eaee4e87a)

 
 
## Reproducing the Code:
- Ensure the requirements.txt are installed
- First run the EDA.ipynb – This contains the exploratory data analysis of the dataset, the processing is done and saved as a csv file one for the Cause - Hierarchy 1 and another for Body Part - Hierarchy 1.
- Then run the cause_train.ipynb followed by bodypart_train.ipynb or vice versa, this will fine tune the bert-base-uncased model and saves the model locally.
- Run the test.ipynb  - This is where we load both the models and predict the labels for the test.csv. and saved the file in a new test.csv file which contains the predicted labels

## FastAPI Integration
sample input:
"punched ee in the face numerous times by person supported., struck or injured by, head"

![Screenshot (474)](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/8b1ac878-1597-4512-bedb-628021a60f04)

![Screenshot (475)](https://github.com/Gowtham58/Multilabel-Classifier-for-Injury/assets/75661938/53b1cb64-fbce-412f-9507-b6247a8233d7)

we get out put as:
```
{
  "answer": [
    "struck or injured by",
    "head"
  ]
}
```


