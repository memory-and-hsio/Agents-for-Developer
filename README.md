## Template for creating your GPT with Streamlit and OpenAI API


### Demo App
![Demo](https://agents-for-developer-grznmgtffjrbc7qoaz7hcw.streamlit.app/)

### Prerequisites
1. Python 3.6 or above
2. An OpenAI API Key ( https://platform.openai.com/apps )


### Steps to run the application
**1. Clone the repository to your local machine:**
```shell
git clone https://github.com/memory-and-hsio/Agents-for-Developer.git
```

**2. Navigate to the project directory:**
```shell
cd Agents-for-Developer
```

**3. Create a virtual environment and activate it:**

On Windows:
```shell
python -m venv myenv
.\myenv\Scripts\activate
```

**3a. Upgrade pip (optional but recommended).**
```shell
pip install --upgrade pip
```

**4. Install the necessary Python packages:**
```shell
pip install -r requirements.txt
```

**5. Create a .env file in the root directory of the project and add your OpenAI API key:**
```shell
echo OPENAI_API_KEY=your-api-key > .env
```
OR

Please replace your-api-key with your actual OpenAI API key.

**6. Run the Streamlit application:**
```shell
streamlit run Greetings.py
```

Open a web browser and navigate to http://localhost:8501 to interact with the application.




License
This project is open source, under the terms of the MIT license.


