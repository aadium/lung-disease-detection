import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk

from functions import predict_disease

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        lbl_img.configure(image=img)
        lbl_img.image = img

        # Predict the disease
        disease = predict_disease(file_path)
        lbl_disease['text'] = "Predicted Disease: " + disease

root = tk.Tk()
root.title("Disease Predictor")

mainframe = ttk.Frame(root, padding="10")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Button to open the file dialog
btn_open = ttk.Button(mainframe, text="Open Image", command=open_file, 
                      style='Big.TButton')

# Create a style
style = ttk.Style()

# Configure a larger font and padding for the 'Big.TButton' style
style.configure('Big.TButton', font=('Arial', 20), padding=10)

# Label to display the image
lbl_img = ttk.Label(mainframe)
lbl_img.grid(column=0, row=1, sticky=(tk.W, tk.E, tk.N, tk.S))

# Label to display the predicted disease
lbl_disease = ttk.Label(mainframe, text="Predicted Disease: ")
lbl_disease.grid(column=0, row=2, sticky=tk.W)

root.mainloop()