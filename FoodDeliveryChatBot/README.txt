Directory structure
===================
backend: Contains Python FastAPI backend code
db: contains the dump of the database. you need to import this into your MySQL db by using MySQL workbench tool
dialogflow_assets: this has training phrases etc. for our intents
frontend: website code

Install these modules
======================

pip install mysql-connector
pip install "fastapi[all]"

OR just run pip install -r backend/requirements.txt to install both in one shot

To start fastapi backend server
================================
1. Go to backend directory in your command prompt
2. Run this command: uvicorn main:app --reload

================================================================================================
ngrok for https tunneling (in dialogueflow fulfillment webhook url requires https, so ngrok is for that)
================================
1. To install ngrok, go to https://ngrok.com/download and install ngrok version that is suitable for your OS
2. Extract the zip file and place ngrok.exe in a folder.
3. Open windows command prompt, go to that folder and run this command: ngrok http 80000

NOTE: ngrok can timeout. you need to restart the session if you see session expired message.

Steps to Fix ngrok Authentication Issue
1. Sign Up for an ngrok Account
Visit the ngrok Sign-Up Page.
Create an account or log in if you already have one.
Verify your account via email.
2. Get Your AuthToken
After signing in, go to the ngrok Authtoken Page.
Copy the provided authtoken from the dashboard.
3. Add the Authtoken to Your ngrok Configuration
Run the following command in your terminal to add the authtoken:

bash
Copy code
ngrok config add-authtoken YOUR_AUTHTOKEN
Replace YOUR_AUTHTOKEN with the token you copied in the previous step.

4. Start the Tunnel
Once the authtoken is configured, you can start an HTTP tunnel using:

bash
Copy code
ngrok http <port>
For example, to expose port 80:

bash
Copy code
ngrok http 80
5. Verify Paid Features (Optional)
If you’re trying to use paid features (e.g., custom domains, IP restrictions), ensure your account is upgraded. Visit the Billing Page for more details.


1) cd C:\Users\Shivayogi\Documents\Personal\CapstoneProjects\ngrok> 

Run ( ./ngrok config add-authtoken 2qUZr0GpL1yknR61bnIBIu8X4uo_4UTu4SzuHLgzojHv1AXpR )
Authtoken saved to configuration file: C:\Users\Shivayogi\AppData\Local/ngrok/ngrok.yml

2) run "./ngrok http 8000" 

Web Interface  http://127.0.0.1:4040  Forwarding https://5baa-2406-7400-104-bf2b-cd94-bad2-ed9e-4a79.ngrok-free.app -> http://localhost:8000   
================================================================================================================================


#MySQL
if you are not able to use the root pwd for mysql workbench .. stop the mysql80 service 
 if the mysql80 service is running.. and on the cmd prompt run 
   "C:\Program Files\MySQL\MySQL Server 8.0\bin>mysqld --console"
on another cmd promt run ".\mysql -u root -p" .. it will work ..
now use the workbench as well ..  (Dont have to restart the service)
================================================================================================

Suggested improvements for the existing UI:

Dashboard Overview


Add KPI cards showing current metrics
Include real-time delivery status counters
Show daily/weekly prediction accuracy rates


Prediction Interface


Add historical trend comparison
Display confidence scores with predictions
Integrate weather and traffic API data
Add map visualization for delivery zones


Monitoring Improvements


Add prediction error tracking
Include model drift detection alerts
Show feature importance graphs


New Features


Restaurant capacity planning tool
Driver allocation optimizer
Cost estimation calculator
Batch prediction interface for multiple orders


UX Enhancements


Add mobile-responsive design
Implement dark mode
Add export functionality
Improve error messages

-------------------------------------------
DB Schema 

Key relationships and purposes:

Users: Customer/restaurant authentication
Orders: Links customers, restaurants, drivers
Deliveries: ML prediction features
PredictionLogs: Model performance tracking
DemandForecasts: Peak demand predictions