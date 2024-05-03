import os
import time
import json
import mysql.connector

import numpy as np
from sklearn.neighbors import KNeighborsClassifier  # Import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier

# Paths to training folders
leak_folder = 'C:/Users/ACER/SVM/MIX/Data set _Leak'
no_leak_folder = 'C:/Users/ACER/SVM/MIX/Data set _No leak'
new_data_folder = 'C:/Users/ACER/SVM/test'

# Function to read JSON files from a folder and return data and labels
def read_json_files_from_folder(folder_path, label):
    data_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                file_content = json.load(file)
                file_content['label'] = label  # Add label to the data
                data_list.append(file_content)
    return data_list

# Record start time
start_time = time.time()

# Read data from 'leak' and 'no leak' folders
leak_data = read_json_files_from_folder(leak_folder, label='leak')
no_leak_data = read_json_files_from_folder(no_leak_folder, label='no leak')

# Concatenate data
data = leak_data + no_leak_data

# Convert data to DataFrame
df = pd.DataFrame(data)

# Remove columns if they exist in the DataFrame
columns_to_remove = ['protected']
existing_columns = df.columns.tolist()

columns_to_drop = [col for col in columns_to_remove if col in existing_columns]
df.drop(columns=columns_to_drop, inplace=True)

# Convert text data to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['payload'].apply(json.dumps))  # Convert each dictionary to a JSON string
y = df['label'].values
print(X_tfidf)
# Convert TF-IDF data to dense numpy arrays
X_tfidf_dense = X_tfidf.toarray()

# Calculate the number of features (n_features)
n_features = X_tfidf_dense.shape[1]

# Calculate the maximum allowed value for n_components
max_n_components = min(n_features, len(np.unique(y)) - 1)

# Use max_n_components to set n_components in LDA
lda = LDA(n_components=max_n_components)

# Apply LDA on the data
X_lda = lda.fit_transform(X_tfidf_dense, y)

# Create a KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as needed

# Train the KNN classifier on the LDA-reduced data
knn_classifier.fit(X_lda, y)

# Read new data from the test folder
new_data = read_json_files_from_folder(new_data_folder, label='unknown')  # 'unknown' label for new data

# Convert new text data to numerical features using the same vectorizer
new_X_tfidf = vectorizer.transform([json.dumps(entry['payload']) for entry in new_data])

# Convert TF-IDF data to dense numpy arrays for new data
new_X_tfidf_dense = new_X_tfidf.toarray()

# Apply LDA on new data
new_X_lda = lda.transform(new_X_tfidf_dense)

# Predict using the trained KNN classifier for new data
new_predictions_knn = knn_classifier.predict(new_X_lda)

print("Predictions for new data (KNN):", new_predictions_knn)

# Use k-fold cross-validation to get predictions on the entire dataset
all_X = X_lda
all_y = df['label'].values

# Use cross_val_predict with KNN
all_predictions_knn = cross_val_predict(knn_classifier, all_X, all_y, cv=5)

# Display the classification report for cross-validation predictions using KNN
classification_rep_knn = classification_report(all_y, all_predictions_knn, target_names=knn_classifier.classes_)
print("Classification Report (Cross-validation - KNN):\n", classification_rep_knn)

# Display the confusion matrix for KNN
confusion_knn = confusion_matrix(all_y, all_predictions_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_knn, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=knn_classifier.classes_, yticklabels=knn_classifier.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - KNN')
plt.show()

# Calculate average accuracy using k-fold cross-validation for KNN
accuracy_scores_knn = cross_val_score(knn_classifier, all_X, all_y, cv=5)
mean_accuracy_knn = np.mean(accuracy_scores_knn)
print("Average Accuracy using k-fold cross-validation (KNN):", mean_accuracy_knn)

# Calculate F1-score for KNN
f1_knn = f1_score(all_y, all_predictions_knn, average='weighted')  # or 'micro', 'macro', 'samples', etc. as needed
print("F1 Score (KNN):", f1_knn)

# Calculate recall for KNN
recall_knn = recall_score(all_y, all_predictions_knn, average='weighted')  # or 'micro', 'macro', 'samples', etc. as needed
print("Recall (KNN):", recall_knn)

# Calculate precision for KNN
precision_knn = precision_score(all_y, all_predictions_knn, average='weighted')  # or 'micro', 'macro', 'samples', etc. as needed
print("Precision (KNN):", precision_knn)
# Calculate the confusion matrix for KNN
confusion_knn = confusion_matrix(all_y, all_predictions_knn)

# Extract elements from the confusion matrix
tn_knn, fp_knn, fn_knn, tp_knn = confusion_knn.ravel()

# Calculate True Positive Rate (TPR) and True Negative Rate (TNR) for KNN
tpr_knn = tp_knn / (tp_knn + fn_knn)
tnr_knn = tn_knn / (tn_knn + fp_knn)

print("True Positive Rate (TPR - KNN):", tpr_knn)
print("True Negative Rate (TNR - KNN):", tnr_knn)

# Calculate False Positive Rate (FPR) and False Negative Rate (FNR) for KNN
fpr_knn = fp_knn / (fp_knn + tn_knn)
fnr_knn = fn_knn / (fn_knn + tp_knn)

print("False Positive Rate (FPR - KNN):", fpr_knn)
print("False Negative Rate (FNR - KNN):", fnr_knn)
# Import missing libraries for ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score

# ... (previous code)

# Tracer la courbe ROC
all_probabilities = cross_val_predict(knn_classifier, all_X, all_y, cv=5, method='predict_proba')
positive_class_probabilities = all_probabilities[:, 1]

fpr, tpr, thresholds = roc_curve(all_y, positive_class_probabilities, pos_label='leak')
roc_auc = roc_auc_score(all_y, positive_class_probabilities)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - KNN')  # Add a title
plt.legend(loc='lower right')
plt.show()

# ... (remaining code)

# Record end time
end_time_knn = time.time()

# Calculate total execution time for KNN
execution_time_knn = end_time_knn - start_time
print("Total Execution Time (KNN): {:.2f} seconds".format(execution_time_knn))
# 1. Add imports
import json
from web3 import Web3
from web3.middleware import geth_poa_middleware

# 2. Add the Web3 provider logic here:
provider_rpc = {
    'development': 'http://localhost:9545',
}
web3 = Web3(Web3.HTTPProvider(provider_rpc['development']))  # Change to correct network
web3.middleware_onion.inject(geth_poa_middleware, layer=0)  # Add this line for Moonbeam network compatibility

# 3. Create variables
account_from = {
    'private_key': '0be4cc97daabdd7c451bb27acb494cacd029525ac4aeb1e6f719b792203df1a0',
    'address': Web3.to_checksum_address('0xf6246dc675597686f778044cd5c05faa9a9673e9'),
}
contract_address =  '0xabEfC51AA58b31E46b00872911020Ca384B92DC6'
print(
    f'Calling the storeData function in contract at address: {contract_address}'
)

# Convert the lowercase address to checksum address using web3.toChecksumAddress from the checksum module

abi = [
    {
      "inputs": [
        {
          "internalType": "string",
          "name": "_zone",
          "type": "string"
        },
        {
          "internalType": "string",
          "name": "_ID_noeud",
          "type": "string"
        },
        {
          "internalType": "uint256",
          "name": "_Date",
          "type": "uint256"
        }
      ],
      "name": "storeData",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_timestamp",
          "type": "uint256"
        }
      ],
      "name": "getDataByDate",
      "outputs": [
        {
          "internalType": "string",
          "name": "",
          "type": "string"
        },
        {
          "internalType": "string",
          "name": "",
          "type": "string"
        },
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        },
        {
          "internalType": "bytes32",
          "name": "",
          "type": "bytes32"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "getAllTransactions",
      "outputs": [
        {
          "components": [
            {
              "internalType": "string",
              "name": "zone",
              "type": "string"
            },
            {
              "internalType": "string",
              "name": "ID_noeud",
              "type": "string"
            },
            {
              "internalType": "uint256",
              "name": "Date",
              "type": "uint256"
            },
            {
              "internalType": "bytes32",
              "name": "dataHash",
              "type": "bytes32"
            }
          ],
          "internalType": "struct earn_project.Data[]",
          "name": "",
          "type": "tuple[]"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    }
  ]
contract = web3.eth.contract(address=contract_address, abi=abi)


# Get the account's transaction count
nonce = web3.eth.get_transaction_count(Web3.to_checksum_address(account_from['address']))


# Iterate through new_predictions and new_data simultaneously
for prediction, entry in zip(new_predictions_knn, new_data):
    if prediction == 'leak':
       
        # Check the structure of the entry dictionary
        if 'zone' in entry and 'signature' in entry and 'Timestamp' in entry:
            # Prepare data for storing in the smart contract
            data_to_store = {
                'zone': entry['zone'],
                'id_noeud': entry['signature'],
                'timestamp': entry['Timestamp']
            }

            # Build and send transaction to call storeData function
            store_data_tx = contract.functions.storeData(data_to_store['zone'], data_to_store['id_noeud'], data_to_store['timestamp']).build_transaction(
                {
                    'from': account_from['address'],
                    'nonce': nonce,
                    'gas': 2000000,
                }
            )
            # Sign and send the transaction
            signed_tx = web3.eth.account.sign_transaction(store_data_tx, private_key=account_from['private_key'])
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            print(f'Tx successful with hash: {tx_receipt.transactionHash.hex()}')
            # Vous pouvez vérifier le statut de la transaction pour savoir si elle a été confirmée
if tx_receipt.status == 1:
    print("La transaction a été confirmée.")
else:
    print("La transaction a échoué ou a été annulée.")

# Si vous voulez vérifier les détails du contrat après l'envoi des données
all_transactions = contract.functions.getAllTransactions().call()
for transaction in all_transactions:
    print(f"Zone : {transaction[0]}, ID du nœud : {transaction[1]}, Timestamp : {transaction[2]}")

def insert_data_into_mysql(predictions, new_data):
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="iteb1234",
            database="pfe",
            charset="utf8"
        )
        cursor = connection.cursor()

        for prediction, entry in zip(predictions, new_data):
            if prediction == 'leak':
                signature = entry['signature']  # Assuming 'signature' key exists in the entry dictionary
                zone = entry['zone']  # Assuming 'zone' key exists in the entry dictionary
                Timestamp = entry['Timestamp']  # Assuming 'Timestamp' key exists in the entry dictionary

                signature_str = json.dumps(signature)  # Convert signature to JSON format

                # Assuming you have a database connection and cursor created
                query = "INSERT INTO leak (signature, zone, Timestamp) VALUES (%s, %s, %s)"  # Use parameterized query
                cursor.execute(query, (signature_str, zone, Timestamp))
        
                connection.commit()

        cursor.close()

    except mysql.connector.Error as e:
        print("Error in MySQL connection:", e)

    finally:
        connection.close()



insert_data_into_mysql(new_predictions_knn, new_data)
print("\nData insertion into MySQL successful")
# Iterate through new_predictions and new_data simultaneously
for prediction, entry in zip(new_predictions_knn, new_data):
    if prediction == 'leak':
        # Récupérer les valeurs de la colonne "payload"
        y_values = entry['payload']  # Assurez-vous que la clé "payload" existe dans entry
        x = np.arange(len(y_values))

        # Tracer le graphique pour chaque prédiction "leak"
        plt.figure(figsize=(8, 4))
        plt.plot(x, y_values)
        plt.title('Graph for Leak Prediction')
        plt.xlabel('X Label')
        plt.ylabel('Y Label')
        plt.grid(True)
        plt.show()

