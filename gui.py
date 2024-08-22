from tkinter import Tk, Canvas, Button, OptionMenu, StringVar, Entry, Label
import pandas as pd
import pickle

# Load model
try:
    with open('used_car_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Load data
combination_df = pd.read_csv('bmyencode.csv')
brands_df = pd.read_csv('brands.csv')
model_df = pd.read_csv('model.csv')
transmission_df = pd.read_csv('Transmission.csv')
year_df = pd.read_csv('year.csv')

# Extract data
model_names = model_df['model'].tolist()
brand_names = brands_df['brand'].tolist()
transmission_types = transmission_df['fuel_type'].tolist()
year = year_df.iloc[:, 1].tolist()
unique_years = sorted(set(year))

# Create encoding dictionaries
brand_encode_dict = dict(zip(brands_df['brand'], brands_df['brand_encode']))
model_encode_dict = dict(zip(model_df['model'], model_df['model_encode']))
transmission_encode_dict = dict(zip(transmission_df['fuel_type'], transmission_df['fuel_type_encode']))

def backend_function():
    try:
        # Validate and encode inputs
        km_traveled = int(km_entry.get())
        selected_brand = brand_var.get()
        selected_model = model_var.get()
        selected_transmission = transmission_var.get()
        selected_year = year_var.get()

        brand_encode_value = brand_encode_dict.get(selected_brand)
        model_encode_value = model_encode_dict.get(selected_model)
        transmission_encode_value = transmission_encode_dict.get(selected_transmission)
        
        encoded_combination = combination_df[
            combination_df['brand_model_and_year_combined'] == (f"{brand_encode_value}_{model_encode_value}_{selected_year}")
        ].values

        if not encoded_combination.size:
            price_label.config(text="No valid combination found.")
            return

        encoded_values = [encoded_combination[0][2], km_traveled, transmission_encode_value]
        
        # Predict and display the price
        price = loaded_model.predict([encoded_values])
        price_label.config(text=f"Price: EGP{price[0]:,.2f}")
    except ValueError:
        price_label.config(text="Please enter valid inputs.")
    except Exception as e:
        price_label.config(text=f"An error occurred: {e}")

# Create GUI
window = Tk()
window.geometry("692x359")
window.configure(bg="#FFFFFF")

canvas = Canvas(window, bg="#FFFFFF", height=359, width=692, bd=0, highlightthickness=0, relief="ridge")
canvas.place(x=0, y=0)

button_1 = Button(text="Submit", borderwidth=0, highlightthickness=0, command=backend_function, relief="flat")
button_1.place(x=285.0, y=303.0, width=121.0, height=40.0)

model_var = StringVar(value="Select Model")
brand_var = StringVar(value="Select Brand")
transmission_var = StringVar(value="Select Transmission")
year_var = StringVar(value="Select Year")

model_dropdown = OptionMenu(window, model_var, *model_names)
model_dropdown.place(x=55.5, y=154.0, width=120.0, height=23.0)

brand_dropdown = OptionMenu(window, brand_var, *brand_names)
brand_dropdown.place(x=285.5, y=188.0, width=120.0, height=23.0)

transmission_dropdown = OptionMenu(window, transmission_var, *transmission_types)
transmission_dropdown.place(x=489.5, y=154.0, width=120.0, height=23.0)

year_dropdown = OptionMenu(window, year_var, *unique_years)
year_dropdown.place(x=55.5, y=240.0, width=120.0, height=23.0)

canvas.create_text(76.0, 139.0, anchor="nw", text="MODEL", fill="#000000", font=("Roboto Regular", 9 * -1))
canvas.create_text(327.0, 174.0, anchor="nw", text="BRAND", fill="#000000", font=("Roboto Regular", 9 * -1))
canvas.create_text(514.0, 139.0, anchor="nw", text="TRANSMISSION", fill="#000000", font=("Roboto Regular", 9 * -1))
canvas.create_text(484.0, 223.0, anchor="nw", text="KILOMETERS TRAVELED", fill="#000000", font=("Roboto Regular", 9 * -1))
canvas.create_text(76.0, 223.0, anchor="nw", text="MODEL YEAR", fill="#000000", font=("Roboto Regular", 9 * -1))
canvas.create_text(291.0, 8.0, anchor="nw", text="Car Price Estimator", fill="#000000", font=("Inter", 24 * -1))

km_entry = Entry(bd=0, bg="#D9D9D9", fg="#000716", highlightthickness=0)
km_entry.place(x=489.5, y=240.0, width=120.0, height=23.0)

# Label to display the estimated price
price_label = Label(window, text="", bg="#FFFFFF", font=("Roboto Regular", 12))
price_label.place(x=285.0, y=260.0, width=200.0, height=30.0)

window.resizable(False, False)
window.mainloop()
