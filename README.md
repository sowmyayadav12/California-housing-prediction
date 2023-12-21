# California-housing-prediction

## Overview
This project focuses on predicting housing prices in California using machine learning techniques. The dataset used is based on the California housing data, and the goal is to build a predictive model to estimate the median house value for districts in California.

## Project Structure

- **data:** Contains the dataset used for training and testing the model.
- **notebooks:** Jupyter notebooks,google collab used for data exploration, model training, and evaluation.
- **scripts:** Any additional scripts used in the project.
- **models:** Saved machine learning models.
- **results:** Any output, visualizations, or results generated during the analysis.
- **requirements.txt:** Dependencies for running the project.

## Getting Started

### Prerequisites
- Python 3.x
- Jupyter Notebook,google collab
- Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn (install dependencies using `pip install -r requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/california-housing-prediction.git
   cd california-housing-prediction
### output screenhots
<img width="915" alt="Screenshot 2023-12-21 131309" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/bf7754b5-d596-4bbd-942e-fedcda4a45c6">

<img width="923" alt="Screenshot 2023-12-21 131328" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/3509b2b8-ac3d-4d47-84ba-69ca5727abd5">
<img width="925" alt="Screenshot 2023-12-21 131350" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/8b726972-b8f5-4c5c-a183-9d359ad058fa">
<img width="922" alt="Screenshot 2023-12-21 131408" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/d13208a4-fbff-403a-8c33-a990de41f84f">
<img width="921" alt="Screenshot 2023-12-21 131425" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/cef5fbd8-461d-414b-9d8f-3a2d448f67ea">
<img width="929" alt="Screenshot 2023-12-21 131446" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/65c283c9-1c03-4985-9070-7da8d86b7e92">
<img width="917" alt="Screenshot 2023-12-21 131504" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/bce18b2e-cf3c-4c5b-abf5-efde90cf18e5">
<img width="933" alt="Screenshot 2023-12-21 131525" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/9b4069c2-902a-4b79-a09f-3a936ddbe5da">
<img width="916" alt="Screenshot 2023-12-21 131541" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/1345a7e7-3fa1-4df4-afb3-5fd14fb944b4">
<img width="933" alt="Screenshot 2023-12-21 131556" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/17cd1afa-6866-4746-ad44-9f719ef467c5">
<img width="936" alt="Screenshot 2023-12-21 131619" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/ca2bdcd3-33a1-44e4-a14d-3089e77c9959">
<img width="924" alt="Screenshot 2023-12-21 131634" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/00be4236-2826-49fc-9e63-39cd2c912210">
<img width="925" alt="Screenshot 2023-12-21 131654" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/0f0e0a45-e33c-4d57-9d62-80622fede092">
<img width="932" alt="Screenshot 2023-12-21 131708" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/ab2b1977-32af-4a55-a3c8-afbd4d510ad6">
<img width="927" alt="Screenshot 2023-12-21 131727" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/10144928-79f6-4020-b2b4-125dc6d83180">
<img width="936" alt="Screenshot 2023-12-21 131746" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/6dce0add-db53-45ca-855a-e59d68a29349">
<img width="925" alt="Screenshot 2023-12-21 131804" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/da2226f7-e6bb-4cb1-b149-7f7a819c9a3a">
<img width="916" alt="Screenshot 2023-12-21 131822" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/f<img width="938" alt="Screenshot 2023-12-21 131842" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/bb288ed6-4c73-4949-8fa4-da72d62e7543">
3fff0a5-2f67-4051-b977-66f2d858fa0c">
<img width="918" alt="Screenshot 2023-12-21 131900" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/acd70921-f6de-4f20-b431-0f900827391d">
<img width="923" alt="Screenshot 2023-12-21 131917" src="https://github.com/sowmyayadav12/California-housing-predi<img width="929" alt="Screenshot 2023-12-21 131930" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/1d40dc29-9de5-48ed-8761-c7ef604efc59">
ction/assets/121338921/a02be6f3-f7eb-4d75-bca7-323db87155dc">
<img width="925" alt="Screenshot 2023-12-21 131946" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/32b7297c-3c9e-41ce-bdbe-9810fa3d1a61">
<img width="936" alt="Screenshot 2023-12-21 132001" src="https://github.com/sowmyayadav12/California-housing-predict<img width="932" alt="Screenshot 2023-12-21 132016" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/2d9183b9-1a6e-4fa1-b552-07d3ab78d751">
ion/asse<img width="926" alt="Screenshot 2023-12-21 132029" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/a4a29a78-3d55-4b9a-9849-fe4cb573bc9f">
ts/121338921/da83cd38-5350-4399-8fce-3dc6400d55b5">
<img width="929" alt="Screenshot 2023-12-21 132045" src="https://github.com/sowmyayadav12/California-housing-prediction/assets/121338921/035104c2-7d70-42f5-8f7d-2aa1dcc6084a">
