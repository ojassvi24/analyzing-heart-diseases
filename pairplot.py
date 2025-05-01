import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """
    Main function to load data and generate relationship plots.
    """
    # Load the dataset
    print("Loading dataset...")
    data = pd.read_csv('cardio_train.csv', sep=';')
    
    # 1. Correlation Matrix Heatmap
    print("Generating correlation matrix heatmap...")
    # Calculate correlation matrix
    correlation_matrix = data.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # 2. Pairplot to visualize relationships between key features
    print("Generating pairplot (this may take a while)...")
    sns.pairplot(data[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'cardio']],
                 hue='cardio', palette='coolwarm')
    plt.suptitle('Pairplot of Features Colored by Cardio Status', y=1.02)
    plt.show()
    
    # 3. Box plots for feature distributions by cardio status
    print("Generating box plots...")
    # Box plot for 'age' grouped by 'cardio'
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cardio', y='age', data=data, palette='coolwarm')
    plt.title('Age Distribution by Cardio Status')
    plt.xlabel('Cardio Status')
    plt.ylabel('Age (days)')
    plt.show()
    
    # Box plot for 'weight' grouped by 'cardio'
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cardio', y='weight', data=data, palette='coolwarm')
    plt.title('Weight Distribution by Cardio Status')
    plt.xlabel('Cardio Status')
    plt.ylabel('Weight (kg)')
    plt.show()
    
    print("Done!")

if __name__ == "__main__":
    main()
