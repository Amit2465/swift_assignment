import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit_shadcn_ui as ui
from st_aggrid import AgGrid, GridOptionsBuilder
import missingno as msno
from local_components import card_container

st.set_page_config(page_title="Assignment", page_icon=":bar_chart:", layout="wide")


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)



def load_data(filename1, filename2):
    data1 = pd.read_csv(f"{filename1}.csv")
    data2 = pd.read_csv(f"{filename2}.csv")    
    data = pd.merge(data1, data2, on='customerEmail', how='inner')
    data.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'], inplace=True)
    return data


def ml_model():
    st.markdown("I have created the model using a Random Forest Classifier. For more details about the model and to access the Python notebook, please check the link below.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:        
        ui.card(title="Model Score", content="94%", description="Random Forest Classifier", key="card1").render()
        ui.link_button(text="Notebook", url="https://github.com/ObservedObserver/streamlit-shadcn-ui", key="link_btn")
   
        

def main():
    data = load_data("customer_transaction_details", "customers_df")
    st.title("Assignment")
    select = ui.tabs(options=['Dataset Overview','EDA', 'Model'], default_value='Overview', key="kanaries")
    if select == "Dataset Overview":
        st.write("There are two data set in the form of CSV files. The first dataset contains transaction details of customers and the second dataset contains customer details. The goal is to analyze the data and build a model to predict fraud in the transactions.")    
        st.write("I had merged the datasets on the customerEmail column to create a single dataset for analysis.")
        st.markdown("---")
        
        st.markdown("## First Ten rows")
        gb = GridOptionsBuilder.from_dataframe(data)
        gb.configure_grid_options(domLayout='autoHeight')
        grid_options = gb.build()
        AgGrid(data.head(10), gridOptions=grid_options, fit_columns_on_grid_load=False )
        st.markdown("---") 
        
        
        st.write(f"There are `{data.shape[0]}` rows and `{data.shape[1]}` columns in the dataset")
        
        
        column_info = pd.DataFrame({
        'Column Name': data.columns,
        'Data Type': [data[col].dtype for col in data.columns]
        })

        with st.expander("Show all columns"):
            st.write(column_info)
        st.markdown("---")         
        st.markdown("## missing values")    
        col19, col20 = st.columns(2)
        with col19: 
            fig, ax = plt.subplots(figsize=(4, 2))
            msno.bar(data, ax=ax, fontsize=8)
            ax.tick_params(axis='x', labelsize=8)
            st.pyplot(fig)
            plt.clf()
            plt.close(fig) 
        with col20:
            st.write('''The bar chart shows that there are no missing values in your dataset. 
                     Each bar corresponds to a column, and since all bars are of full height, 
                     it confirms that every column is fully populated. This indicates that no data
                     cleaning is needed for missing values, allowing you to proceed directly with
                     analysis or modeling.                     
                     ''')    
        
        st.markdown("## Summary statistics of numerical columns")
        st.write(data.describe())
        st.markdown("---") 
        
    
    if select == "EDA":        
        st.markdown("### Histograms of Numerical Features")
        col1, col2, col3= st.columns(3)
        with col1:
            with card_container(key= "hist1"):
                # Create histogram
                plt.figure(figsize=(10, 6))
                data['No_Transactions'].hist(bins=20, figsize=(10, 6))
                plt.suptitle('Histogram of No_transaction')
                
                # Display the plot in Streamlit
                st.pyplot(plt)
        
        with col2:
            with card_container(key="hist2"):
                # Create histogram
                plt.figure(figsize=(10, 6))
                data['No_Orders'].hist(bins=20, figsize=(10, 6))
                plt.suptitle('Histogram of No_Orders')
                
                # Display the plot in Streamlit
                st.pyplot(plt)
                
        with col3:
            with card_container(key="hist3"):
                # Create histogram
                plt.figure(figsize=(10, 6))
                data['No_Payments'].hist(bins=20, figsize=(10, 6))
                plt.suptitle('Histogram of No_Payments')
                
                # Display the plot in Streamlit
                st.pyplot(plt) 
        st.markdown("---")        
                
                
        st.markdown("### Box plot for numerical columns")   
        col4, col5 = st.columns(2)     
        
               
        with col4:
            with card_container(key="chart2"):
                numerical_features = ['No_Transactions', 'No_Orders', 'No_Payments']
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=data[numerical_features])
                plt.title('Boxplots of Numerical Features')
                st.pyplot(plt) 
        with col5:
            with card_container(key="inf1"):
                st.write('''- Since there are outliers in the No_Payments and No_Orders columns,
                         we used the Interquartile Range (IQR) method to remove the outliers from these 
                        columns.''')
                st.write('''- Given that the outliers account for only about `3.3%` of the total data `27` out of `819` rows and without a clear understanding of their origin or
                         impact, I’ve decided to remove them. This choice is justified because the 
                         outliers represent a small portion of the dataset, and their presence could 
                         potentially skew the analysis or degrade model performance. However, it's
                         crucial to document this decision and be mindful of any effects it might
                         have on the overall analysis.''')
    
        st.markdown("---")
        count_data = data['paymentMethodRegistrationFailure'].value_counts(normalize=True).reset_index()
        count_data.columns = ['paymentMethodRegistrationFailure', 'percentage']
        count_data['count'] = data['paymentMethodRegistrationFailure'].value_counts().values

        
        st.markdown("### Count Plots of Categorical Features")
        col7, col8, col9 = st.columns(3)
        with col7:
            with card_container(key="chart3"): 
                st.vega_lite_chart(count_data, {
                    'mark': {'type': 'bar', 'tooltip': True, 'fill': 'rgb(173, 250, 29)', 'cornerRadiusEnd': 4 },
                    'encoding': {
                        'x': {'field': 'paymentMethodRegistrationFailure', 'type': 'ordinal', 'title': 'Payment Method Registration Failure'},
                        'y': {'field': 'count', 'type': 'quantitative', 'title': 'Count'},
                    },
                    'layer': [
                        {  # Bar layer
                            'mark': 'bar',
                        },
                        {  # Text layer for percentages
                            'mark': {
                                'type': 'text',
                                'dy': -10,  # Position the text above the bars
                                'color': 'black'
                            },
                            'encoding': {
                                'text': {
                                    'field': 'percentage',
                                    'type': 'quantitative',
                                    'format': '.1%',  # Format as percentage
                                },
                                'x': {'field': 'paymentMethodRegistrationFailure', 'type': 'ordinal'},
                                'y': {'field': 'count', 'type': 'quantitative'},
                            }
                        }
                    ]
                }, use_container_width=True)
                
                
        with col8:
            count_data1 = data['paymentMethodType'].value_counts(normalize=True).reset_index()
            count_data1.columns = ['paymentMethodType', 'percentage']
            count_data1['count'] = data['paymentMethodType'].value_counts().values
            with card_container(key="chart3"): 
                st.vega_lite_chart(count_data1, {
                    'mark': {'type': 'bar', 'tooltip': True, 'fill': 'rgb(173, 250, 29)', 'cornerRadiusEnd': 4 },
                    'encoding': {
                        'x': {'field': 'paymentMethodType', 'type': 'ordinal', 'title': 'Payment Method Type'},
                        'y': {'field': 'count', 'type': 'quantitative', 'title': 'Count'},
                    },
                    'layer': [
                        {  # Bar layer
                            'mark': 'bar',
                        },
                        {  # Text layer for percentages
                            'mark': {
                                'type': 'text',
                                'dy': -10,  # Position the text above the bars
                                'color': 'black'
                            },
                            'encoding': {
                                'text': {
                                    'field': 'percentage',
                                    'type': 'quantitative',
                                    'format': '.1%',  # Format as percentage
                                },
                                'x': {'field': 'paymentMethodType', 'type': 'ordinal'},
                                'y': {'field': 'count', 'type': 'quantitative'},
                            }
                        }
                    ]
                }, use_container_width=True) 
                
        with col9:
            count_data2 = data['transactionFailed'].value_counts(normalize=True).reset_index()
            count_data2.columns = ['transactionFailed', 'percentage']
            count_data2['count'] = data['transactionFailed'].value_counts().values
            with card_container(key="chart3"): 
                st.vega_lite_chart(count_data2, {
                    'mark': {'type': 'bar', 'tooltip': True, 'fill': 'rgb(173, 250, 29)', 'cornerRadiusEnd': 4 },
                    'encoding': {
                        'x': {'field': 'transactionFailed', 'type': 'ordinal', 'title': 'Transaction Failed'},
                        'y': {'field': 'count', 'type': 'quantitative', 'title': 'Count'},
                    },
                    'layer': [
                        {  # Bar layer
                            'mark': 'bar',
                        },
                        {  # Text layer for percentages
                            'mark': {
                                'type': 'text',
                                'dy': -10,  # Position the text above the bars
                                'color': 'black'
                            },
                            'encoding': {
                                'text': {
                                    'field': 'percentage',
                                    'type': 'quantitative',
                                    'format': '.1%',  # Format as percentage
                                },
                                'x': {'field': 'transactionFailed', 'type': 'ordinal'},
                                'y': {'field': 'count', 'type': 'quantitative'},
                            }
                        }
                    ]
                }, use_container_width=True) 
                
        col10, col11 =st.columns(2)
        with col10:
            count_data3 = data['orderState'].value_counts(normalize=True).reset_index()
            count_data3.columns = ['orderState', 'percentage']
            count_data3['count'] = data['orderState'].value_counts().values
            with card_container(key="chart3"): 
                st.vega_lite_chart(count_data3, {
                    'mark': {'type': 'bar', 'tooltip': True, 'fill': 'rgb(173, 250, 29)', 'cornerRadiusEnd': 4 },
                    'encoding': {
                        'x': {'field': 'orderState', 'type': 'ordinal', 'title': 'Order State'},
                        'y': {'field': 'count', 'type': 'quantitative', 'title': 'Count'},
                    },
                    'layer': [
                        {  # Bar layer
                            'mark': 'bar',
                        },
                        {  # Text layer for percentages
                            'mark': {
                                'type': 'text',
                                'dy': -10,  # Position the text above the bars
                                'color': 'black'
                            },
                            'encoding': {
                                'text': {
                                    'field': 'percentage',
                                    'type': 'quantitative',
                                    'format': '.1%',  # Format as percentage
                                },
                                'x': {'field': 'orderState', 'type': 'ordinal'},
                                'y': {'field': 'count', 'type': 'quantitative'},
                            }
                        }
                    ]
                }, use_container_width=True) 
        
        
        with col11:
            count_data4 = data['paymentMethodProvider'].value_counts(normalize=True).reset_index()
            count_data4.columns = ['paymentMethodProvider', 'percentage']
            count_data4['count'] = data['paymentMethodProvider'].value_counts().values
            with card_container(key="chart3"): 
                st.vega_lite_chart(count_data4, {
                    'mark': {'type': 'bar', 'tooltip': True, 'fill': 'rgb(173, 250, 29)', 'cornerRadiusEnd': 4 },
                    'encoding': {
                        'x': {'field': 'paymentMethodProvider', 'type': 'ordinal', 'title': 'Order State'},
                        'y': {'field': 'count', 'type': 'quantitative', 'title': 'Count'},
                    },
                    'layer': [
                        {  # Bar layer
                            'mark': 'bar',
                        },
                        {  # Text layer for percentages
                            'mark': {
                                'type': 'text',
                                'dy': -10,  # Position the text above the bars
                                'color': 'black'
                            },
                            'encoding': {
                                'text': {
                                    'field': 'percentage',
                                    'type': 'quantitative',
                                    'format': '.1%',  # Format as percentage
                                },
                                'x': {'field': 'paymentMethodProvider', 'type': 'ordinal'},
                                'y': {'field': 'count', 'type': 'quantitative'},
                            }
                        }
                    ]
                }, use_container_width=True)
        
        with card_container(key="card_info"):
            st.markdown("### Payment Insights")
            st.write("""
            1. <b>Payment Method Registration Failure</b>
            <br>- <b>Insight:</b> `10%` of payment attempts fail.
            <br>- <b>Implication:</b> This high failure rate could indicate issues with the payment process or user experience, potentially impacting customer satisfaction and revenue.
            
            2. <b>Payment Method Type</b>
            <br>- <b>Card:</b> `79%`
            <br>- <b>Bitcoin:</b> `8.1%`
            <br>- <b>Paypal:</b> `6.9%`
            <br>- <b>Apple Pay:</b> `5.9%`
            <br>- <b>Insight:</b> The majority of customers prefer card payments. Understanding these preferences can help tailor marketing strategies and establish partnerships with payment providers.
            
            3. <b>Payment Method Provider</b>
            <br>- <b>JCB 16 digit:</b> `21.1%`
            <br>- <b>VISA 16 digit:</b> `20.8%`
            <br>- <b>Maestro:</b> `13%`
            <br>- <b>Voyager:</b> `10.2%`
            <br>- <b>American Express:</b> `7.8%`
            <br>- <b>VISA 13 digit:</b> `7.2%`
            <br>- <b>Diners Club / Carte Blanche:</b> `5.1%`
            <br>- <b>JCB 15 digit:</b> `5.7%`
            <br>- <b>Discover:</b> `5.2%`
            <br>- <b>Mastercard:</b> `3.9%`
            <br>- <b>Insight:</b> The distribution of payment providers varies, affecting decisions related to processing fees, security measures, and partnerships.
            
            4. <b>Transaction Failed</b>
            <br>- <b>Insight:</b> `24%` of transactions fail.
            <br>- <b>Implication:</b> A high failure rate might suggest technical problems, potential fraud, or issues with payment processing, all of which could significantly impact the business.
            
            5. <b>Order State</b>
            <br>- <b>Insight:</b> `15.3%`of orders are either failed or pending.
            <br>- <b>Implication:</b> A high rate of failed or pending orders may indicate inefficiencies in order processing and fulfillment, highlighting areas for improvement.
            """, unsafe_allow_html=True)

        
                
        st.markdown("---")
        st.markdown('### Correlation Analysis')
        
        col13, col14 = st.columns([1.4,1])
        with col13:
            with card_container(key="heatmap1"):
                # Correlation matrix
                correlation_matrix =data[["No_Transactions","No_Orders","No_Payments",
                                                        "paymentMethodRegistrationFailure",
                                                        "Fraud","transactionAmount","transactionFailed"]].corr()

                # Heatmap of correlation matrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True)
                plt.title('Correlation Matrix')
                st.pyplot(plt)
        with col14:
            pass
        with card_container(key="info2"):
            st.markdown("### Correlation Matrix Insights")   
            st.markdown("""
                            <b>No_Transactions:</b>
                            <br>- Strongly positively correlated with No_Orders `0.906` and moderately correlated with transactionFailed `0.132`.
                            <br>- Weakly correlated with other variables.

                            <b>No_Orders:</b>
                            <br>- Similar to No_Transactions, it's strongly positively correlated with No_Transactions `0.906`.
                            <br>- Moderately correlated with transactionAmount `0.178` and weakly correlated with other variables.

                            <b>No_Payments:</b>
                            <br>- Weakly correlated with No_Transactions `0.357` and No_Orders `0.448`.
                            <br>- No strong correlation with other variables.

                            <b>Fraud:</b>
                            <br>- Very weakly correlated with other variables, except for a moderate correlation with transactionAmount `0.283`.

                            <b>paymentMethodRegistrationFailure:</b>
                            <br>- Weakly correlated with No_Transactions `0.182`.
                            <br>- Very weakly correlated with other variables.

                            <b>transactionAmount:</b>
                            <br>- Moderately correlated with Fraud `0.283`, No_Orders `0.178`, and weakly correlated with No_Transactions `0.102`.

                            <b>transactionFailed:</b>
                            <br>- Weakly correlated with No_Transactions `0.132`.
                            <br>- Very weakly correlated with other variables.
                            """, unsafe_allow_html=True)

        st.markdown("---")
                         
        
        
        st.markdown("</br>", unsafe_allow_html=True)
        st.markdown('### Transaction Metrics and Fraud Analysis')
        with card_container(key="TM1"):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            # Plot for No_Transactions
            sns.countplot(data=data, x='No_Transactions', hue='Fraud', ax=axes[0])
            axes[0].set_title('No_Transactions vs Fraud')
            axes[0].set_xlabel('No_Transactions')
            axes[0].set_ylabel('Count')

            # Plot for No_Orders
            sns.countplot(data=data, x='No_Orders', hue='Fraud', ax=axes[1])
            axes[1].set_title('No_Orders vs Fraud')
            axes[1].set_xlabel('No_Orders')
            axes[1].set_ylabel('Count')

            # Plot for No_Payments
            sns.countplot(data=data, x='No_Payments', hue='Fraud', ax=axes[2])
            axes[2].set_title('No_Payments vs Fraud')
            axes[2].set_xlabel('No_Payments')
            axes[2].set_ylabel('Count')

            plt.tight_layout()
            st.pyplot(plt)
        
        with card_container(key="info1"):
            st.write("""
                     1. <b>No. of Transactions:</b>
                     </br>- If the number of transactions is greater than 8 then the customer is definitely a fraud.
                     2. <b>No. of Orders:</b>
                     </br>- If the number of orders is greater than 5, then the customer is definitely a fraud.
                     3. <b>No. of Payments:</b>
                     </br>- If the number of payments is greater than 4, then the customer is definitely a fraud.
                     """, unsafe_allow_html=True)    
    
    
    if select ==  "Model":
        ml_model()            
                
        
        
            
                    
            
                    
            
            
    

if __name__ == "__main__":
    main()