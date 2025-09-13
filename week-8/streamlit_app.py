"""
COVID-19 Research Analysis Project
Complete implementation covering data loading, exploration, cleaning, analysis, and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class COVID19ResearchAnalyzer:
    """Main class for analyzing COVID-19 research metadata"""
    
    def __init__(self, file_path=None):
        self.df = None
        self.df_cleaned = None
        self.file_path = file_path
        
    def load_data(self, file_path=None):
        """Part 1: Load the metadata.csv file"""
        if file_path:
            self.file_path = file_path
            
        try:
            print("Loading data...")
            self.df = pd.read_csv(self.file_path)
            print(f"✓ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return False
    
    def explore_data(self):
        """Part 1: Basic data exploration"""
        if self.df is None:
            print("Please load data first!")
            return
        
        print("\n" + "="*50)
        print("DATA EXPLORATION RESULTS")
        print("="*50)
        
        # 1. First few rows
        print("\n1. First 5 rows of the dataset:")
        print(self.df.head())
        
        # 2. DataFrame dimensions
        print(f"\n2. Dataset dimensions: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        
        # 3. Column information
        print("\n3. Column data types:")
        print(self.df.dtypes)
        
        # 4. Missing values
        print("\n4. Missing values per column:")
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df)) * 100
        })
        print(missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False))
        
        # 5. Basic statistics
        print("\n5. Basic statistics for numerical columns:")
        print(self.df.describe())
        
        return self.df.info()
    
    def clean_data(self):
        """Part 2: Data cleaning and preparation"""
        if self.df is None:
            print("Please load data first!")
            return
        
        print("\n" + "="*50)
        print("DATA CLEANING PROCESS")
        print("="*50)
        
        # Create a copy for cleaning
        self.df_cleaned = self.df.copy()
        
        # 1. Handle missing values
        print("\n1. Handling missing values...")
        
        # Fill missing titles with "No Title"
        self.df_cleaned['title'] = self.df_cleaned['title'].fillna('No Title')
        
        # Fill missing abstracts with empty string
        self.df_cleaned['abstract'] = self.df_cleaned['abstract'].fillna('')
        
        # Fill missing journals with "Unknown"
        self.df_cleaned['journal'] = self.df_cleaned['journal'].fillna('Unknown')
        
        # Fill missing authors with "Unknown"
        self.df_cleaned['authors'] = self.df_cleaned['authors'].fillna('Unknown')
        
        print("   ✓ Missing values handled")
        
        # 2. Convert date columns
        print("\n2. Converting date columns...")
        try:
            self.df_cleaned['publish_time'] = pd.to_datetime(self.df_cleaned['publish_time'], errors='coerce')
            print("   ✓ Date conversion completed")
        except:
            print("   ✗ Date conversion failed - column might not exist")
        
        # 3. Extract year from publication date
        print("\n3. Extracting publication year...")
        try:
            self.df_cleaned['year'] = self.df_cleaned['publish_time'].dt.year
            # Filter out invalid years
            self.df_cleaned = self.df_cleaned[(self.df_cleaned['year'] >= 2019) & (self.df_cleaned['year'] <= 2024)]
            print(f"   ✓ Year extracted. Data filtered to 2019-2024: {len(self.df_cleaned)} records")
        except:
            print("   ✗ Year extraction failed")
        
        # 4. Create word count columns
        print("\n4. Creating additional features...")
        self.df_cleaned['title_word_count'] = self.df_cleaned['title'].str.split().str.len()
        self.df_cleaned['abstract_word_count'] = self.df_cleaned['abstract'].str.split().str.len()
        print("   ✓ Word count features created")
        
        print(f"\n✓ Cleaning complete! Cleaned dataset: {self.df_cleaned.shape[0]} rows × {self.df_cleaned.shape[1]} columns")
        
        return self.df_cleaned
    
    def analyze_data(self):
        """Part 3: Data analysis"""
        if self.df_cleaned is None:
            print("Please clean data first!")
            return
        
        print("\n" + "="*50)
        print("DATA ANALYSIS RESULTS")
        print("="*50)
        
        results = {}
        
        # 1. Papers by year
        print("\n1. Papers by publication year:")
        papers_by_year = self.df_cleaned['year'].value_counts().sort_index()
        print(papers_by_year)
        results['papers_by_year'] = papers_by_year
        
        # 2. Top journals
        print("\n2. Top 10 journals publishing COVID-19 research:")
        top_journals = self.df_cleaned['journal'].value_counts().head(10)
        print(top_journals)
        results['top_journals'] = top_journals
        
        # 3. Most frequent words in titles
        print("\n3. Most frequent words in titles:")
        all_titles = ' '.join(self.df_cleaned['title'].dropna())
        # Remove common words
        stop_words = {'the', 'of', 'and', 'in', 'to', 'for', 'with', 'a', 'on', 'from', 'by', 'an', 'as', 'at', 'is'}
        words = [word.lower() for word in all_titles.split() if word.lower() not in stop_words and len(word) > 3]
        word_freq = pd.Series(words).value_counts().head(20)
        print(word_freq)
        results['word_freq'] = word_freq
        
        # 4. Publication sources
        print("\n4. Distribution by source:")
        if 'source_x' in self.df_cleaned.columns:
            source_dist = self.df_cleaned['source_x'].value_counts().head(10)
            print(source_dist)
            results['source_dist'] = source_dist
        
        return results
    
    def create_visualizations(self, results):
        """Part 3: Create visualizations"""
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Publications over time
        ax1 = plt.subplot(2, 3, 1)
        if 'papers_by_year' in results:
            results['papers_by_year'].plot(kind='bar', ax=ax1, color='steelblue')
            ax1.set_title('COVID-19 Publications by Year', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Number of Papers')
            ax1.grid(True, alpha=0.3)
        
        # 2. Top journals
        ax2 = plt.subplot(2, 3, 2)
        if 'top_journals' in results:
            results['top_journals'].plot(kind='barh', ax=ax2, color='coral')
            ax2.set_title('Top 10 Publishing Journals', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Number of Papers')
            ax2.set_ylabel('')
            ax2.grid(True, alpha=0.3)
        
        # 3. Word cloud
        ax3 = plt.subplot(2, 3, 3)
        all_titles = ' '.join(self.df_cleaned['title'].dropna())
        if all_titles:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
            ax3.imshow(wordcloud, interpolation='bilinear')
            ax3.set_title('Word Cloud of Paper Titles', fontsize=14, fontweight='bold')
            ax3.axis('off')
        
        # 4. Source distribution
        ax4 = plt.subplot(2, 3, 4)
        if 'source_dist' in results:
            results['source_dist'].plot(kind='pie', ax=ax4, autopct='%1.1f%%')
            ax4.set_title('Distribution by Source', fontsize=14, fontweight='bold')
            ax4.set_ylabel('')
        
        # 5. Abstract word count distribution
        ax5 = plt.subplot(2, 3, 5)
        self.df_cleaned['abstract_word_count'].hist(bins=50, ax=ax5, color='green', alpha=0.7)
        ax5.set_title('Distribution of Abstract Word Counts', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Word Count')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)
        
        # 6. Top words frequency
        ax6 = plt.subplot(2, 3, 6)
        if 'word_freq' in results:
            results['word_freq'].head(15).plot(kind='bar', ax=ax6, color='purple')
            ax6.set_title('Top 15 Most Frequent Words in Titles', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Words')
            ax6.set_ylabel('Frequency')
            ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('covid19_analysis_visualizations.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualizations saved as 'covid19_analysis_visualizations.png'")
        plt.show()
        
        return fig

def create_streamlit_app():
    """Part 4: Create Streamlit application"""
    st.set_page_config(page_title="COVID-19 Research Analysis", layout="wide")
    
    # Title and description
    st.title("🦠 COVID-19 Research Analysis Dashboard")
    st.markdown("""
    This dashboard analyzes the CORD-19 dataset metadata to provide insights into COVID-19 research trends.
    Upload the metadata.csv file to begin the analysis.
    """)
    
    # Sidebar for file upload and controls
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Choose metadata.csv file", type=['csv'])
    
    if uploaded_file is not None:
        # Initialize analyzer
        analyzer = COVID19ResearchAnalyzer()
        
        # Load data
        df = pd.read_csv(uploaded_file)
        analyzer.df = df
        
        # Clean data
        analyzer.df_cleaned = analyzer.clean_data()
        
        # Sidebar filters
        st.sidebar.header("🔧 Filters")
        
        # Year filter
        if 'year' in analyzer.df_cleaned.columns:
            year_range = st.sidebar.slider(
                "Select Year Range",
                int(analyzer.df_cleaned['year'].min()),
                int(analyzer.df_cleaned['year'].max()),
                (int(analyzer.df_cleaned['year'].min()), int(analyzer.df_cleaned['year'].max()))
            )
            
            # Filter data based on year
            filtered_df = analyzer.df_cleaned[
                (analyzer.df_cleaned['year'] >= year_range[0]) & 
                (analyzer.df_cleaned['year'] <= year_range[1])
            ]
        else:
            filtered_df = analyzer.df_cleaned
        
        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Visualizations", "🔍 Data Sample", "📋 Statistics"])
        
        with tab1:
            st.header("Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Papers", f"{len(filtered_df):,}")
            with col2:
                st.metric("Unique Journals", f"{filtered_df['journal'].nunique():,}")
            with col3:
                st.metric("Date Range", f"{filtered_df['year'].min()}-{filtered_df['year'].max()}")
            with col4:
                st.metric("Avg Abstract Length", f"{filtered_df['abstract_word_count'].mean():.0f} words")
            
            # Key insights
            st.subheader("📌 Key Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top 5 Journals**")
                top_journals = filtered_df['journal'].value_counts().head(5)
                st.bar_chart(top_journals)
            
            with col2:
                st.write("**Publications by Year**")
                papers_by_year = filtered_df['year'].value_counts().sort_index()
                st.line_chart(papers_by_year)
        
        with tab2:
            st.header("Data Visualizations")
            
            # Word cloud
            st.subheader("📝 Word Cloud of Paper Titles")
            all_titles = ' '.join(filtered_df['title'].dropna())
            if all_titles:
                fig, ax = plt.subplots(figsize=(12, 6))
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            
            # Distribution plots
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Abstract Word Count Distribution")
                fig, ax = plt.subplots(figsize=(8, 5))
                filtered_df['abstract_word_count'].hist(bins=30, ax=ax, color='steelblue', alpha=0.7)
                ax.set_xlabel('Word Count')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                st.subheader("🏆 Top Publishing Sources")
                if 'source_x' in filtered_df.columns:
                    source_dist = filtered_df['source_x'].value_counts().head(10)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    source_dist.plot(kind='barh', ax=ax, color='coral')
                    ax.set_xlabel('Number of Papers')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        
        with tab3:
            st.header("Data Sample")
            
            # Sample size selector
            sample_size = st.slider("Select sample size", 5, 100, 10)
            
            # Display columns selector
            columns = st.multiselect(
                "Select columns to display",
                filtered_df.columns.tolist(),
                default=['title', 'authors', 'journal', 'year', 'abstract_word_count']
            )
            
            if columns:
                st.dataframe(filtered_df[columns].head(sample_size))
            
            # Download option
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Cleaned Data as CSV",
                data=csv,
                file_name='covid19_cleaned_data.csv',
                mime='text/csv'
            )
        
        with tab4:
            st.header("Statistical Summary")
            
            # Basic statistics
            st.subheader("📊 Numerical Columns Statistics")
            st.dataframe(filtered_df.describe())
            
            # Missing values
            st.subheader("❓ Missing Values Analysis")
            missing_data = pd.DataFrame({
                'Column': filtered_df.columns,
                'Missing Count': filtered_df.isnull().sum(),
                'Missing %': (filtered_df.isnull().sum() / len(filtered_df)) * 100
            })
            st.dataframe(missing_data[missing_data['Missing Count'] > 0].sort_values('Missing %', ascending=False))
            
            # Data types
            st.subheader("🔤 Data Types")
            dtype_df = pd.DataFrame({
                'Column': filtered_df.columns,
                'Data Type': filtered_df.dtypes.astype(str)
            })
            st.dataframe(dtype_df)
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload the metadata.csv file from the CORD-19 dataset to begin analysis.")
        
        st.markdown("""
        ### 📚 About the CORD-19 Dataset
        
        The COVID-19 Open Research Dataset (CORD-19) is a resource of over 1,000,000 scholarly articles about COVID-19, 
        SARS-CoV-2, and related coronaviruses. This dataset is prepared by the White House and a coalition of leading 
        research groups.
        
        ### 🎯 Analysis Features
        
        This dashboard provides:
        - **Data Exploration**: Basic statistics and data structure
        - **Temporal Analysis**: Research trends over time
        - **Journal Analysis**: Top publishing journals
        - **Text Analysis**: Word frequency and word clouds
        - **Interactive Filtering**: Filter by year and other parameters
        - **Data Export**: Download cleaned and filtered data
        
        ### 🚀 Getting Started
        
        1. Download the metadata.csv file from the CORD-19 dataset
        2. Upload it using the file uploader in the sidebar
        3. Explore the different tabs for various analyses
        4. Use filters to focus on specific time periods
        5. Download the cleaned data for further analysis
        """)

# Main execution function
def main():
    """Main function to run the analysis"""
    print("="*60)
    print("COVID-19 RESEARCH ANALYSIS PROJECT")
    print("="*60)
    
    # For command-line execution
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'streamlit':
        # Run Streamlit app
        create_streamlit_app()
    else:
        # Run console analysis
        print("\nPlease provide the path to metadata.csv file:")
        file_path = input("File path: ").strip()
        
        # Initialize analyzer
        analyzer = COVID19ResearchAnalyzer(file_path)
        
        # Part 1: Load and explore data
        print("\n" + "="*60)
        print("PART 1: DATA LOADING AND EXPLORATION")
        print("="*60)
        
        if analyzer.load_data():
            analyzer.explore_data()
        else:
            print("Failed to load data. Please check the file path.")
            return
        
        # Part 2: Clean data
        print("\n" + "="*60)
        print("PART 2: DATA CLEANING AND PREPARATION")
        print("="*60)
        
        analyzer.clean_data()
        
        # Part 3: Analyze data
        print("\n" + "="*60)
        print("PART 3: DATA ANALYSIS AND VISUALIZATION")
        print("="*60)
        
        results = analyzer.analyze_data()
        
        # Create visualizations
        if results:
            analyzer.create_visualizations(results)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("\nTo run the Streamlit app, use:")
        print("streamlit run script_name.py")

if __name__ == "__main__":
    # For Streamlit deployment
    create_streamlit_app()
    
    # Uncomment below for console execution
    # main()