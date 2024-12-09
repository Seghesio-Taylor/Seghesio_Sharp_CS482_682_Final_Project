import decision_tree_xgboost as dt
import logistic_regression as logreg
import residual_neural_network as rnn

def main():
    while True:
        print("\nAI Model Execution Menu:")
        print("1. Run XGBoosted Decision Tree Model")
        print("2. Run Logistic Regression Model")
        print("3. Run Residual Neural Network")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            print("\nRunning XGBoosted Decision Tree Model...")
            dt.main()
            print("\nReturned to Main Menu.")
        elif choice == '2':
            print("\nRunning Logistic Regression Model...")
            logreg.main()
            print("\nReturned to Main Menu.")
        elif choice == '3':
            print("\nRunning Residual Neural Network...")
            rnn.main()
            print("\nReturned to Main Menu.")
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice enter a number between 1-4.")


if __name__ == '__main__':
    main()
