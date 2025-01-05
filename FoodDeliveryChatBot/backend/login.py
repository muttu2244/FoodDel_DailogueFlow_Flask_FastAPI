from flask import Flask, render_template, request, redirect, url_for, flash
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Dummy OTP for demonstration
DUMMY_OTP = '123456'

app = Flask(__name__, template_folder="../templates")
print(app.jinja_loader.searchpath)
from flask import current_app

# Clear the Jinja2 template cache
with app.app_context():
    app.jinja_env.cache.clear()  # Clear Jinja2 template cache



@app.route('/', methods=['GET', 'POST'])
def landing_page():
    error = None
    user_type = request.form.get('user_type', 'customer')
    phone = request.form.get('phone', '')
    otp = request.form.get('otp', '')
    email = request.form.get('email', '')
    password = request.form.get('password', '')
    show_otp = request.form.get('show_otp', 'false') == 'true'

    if request.method == 'POST':
        if user_type == 'customer':
            if not show_otp and len(phone) == 10:
                return render_template('landing_page.html', user_type=user_type, phone=phone, show_otp=True)
            elif show_otp:
                if otp == DUMMY_OTP:
                    return redirect('http://127.0.0.1:8000')
                else:
                    error = f"Invalid OTP. Use: {DUMMY_OTP}"
            else:
                error = 'Please enter a valid 10-digit phone number'
        elif user_type == 'restaurant':
            # Redirect for restaurant login
            return redirect('http://127.0.0.1:8888')

    return render_template('landing_page.html', user_type=user_type, phone=phone, otp=otp,
                           email=email, password=password, show_otp=show_otp, error=error)

if __name__ == '__main__':
    app.run(debug=True)
