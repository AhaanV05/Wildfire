import pandas as pd

def fill_null_with_zero(input_csv_path, output_csv_path):
    """
    Reads a CSV file, replaces all null/NaN values with 0,
    and saves the modified data to a new CSV file.

    Args:
        input_csv_path (str): The path to the input CSV file.
        output_csv_path (str): The path to save the modified CSV file.
    """
    try:
        df = pd.read_csv(input_csv_path)
        df_filled = df.fillna(0)
        df_filled.to_csv(output_csv_path, index=False)
        print(f"Null values in '{input_csv_path}' replaced with 0 and saved to '{output_csv_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
input_file = r'C:\Users\ahaan\CODE\Wildfire\Threat_Predictor\real_features_with_ros.csv'  # Replace with your input CSV file name
output_file = 'Cleaned_ros_features.csv' # Replace with your desired output CSV file name
fill_null_with_zero(input_file, output_file)