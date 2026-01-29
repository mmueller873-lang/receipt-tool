# ============================================================
# MAIN RENDER (Persistent)
# ============================================================

if st.session_state.master_results_df is not None:
    df = st.session_state.master_results_df
    
    # Normalize Data
    df['total'] = pd.to_numeric(df['total'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['display_date'] = df['date'].dt.strftime('%Y-%m-%d')
    df = df.sort_values(by=['date', 'total'], ascending=[False, False])

    # 1. DOWNLOAD BUTTON
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        
        # DYNAMIC COLUMN SELECTION: Only export columns that exist in the dataframe
        desired_summary_cols = ['filename', 'date', 'vendor', 'total', 'category', 'description']
        summary_cols_to_export = [c for c in desired_summary_cols if c in df.columns]
        
        # Sheet 1: Summary
        df.to_excel(writer, index=False, columns=summary_cols_to_export, sheet_name='Summary')
        
        # Sheet 2: Line Items (IF EXISTS)
        if 'items' in df.columns:
            items_list = []
            # Note: Changed from iterrows() to iterrows()
            for index, row in df.iterrows():
                if isinstance(row['items'], list):
                    for item in row['items']:
                        items_list.append({
                            'Parent_File': row['filename'],
                            'Item_Name': item.get('name'),
                            'Qty': item.get('qty'),
                            'Price': item.get('price')
                        })
            
            if items_list:
                items_df = pd.DataFrame(items_list)
                items_df.to_excel(writer, index=False, sheet_name='Line Items')
        
        workbook = writer.book
        # Summary Sheet Formatting
        worksheet = writer.sheets['Summary']
        
        currency_fmt = workbook.add_format({'num_format': '$#,##0.00', 'font_name': 'Arial'})
        date_fmt = workbook.add_format({'num_format': 'yyyy-mm-dd', 'font_name': 'Arial'})
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#2E86C1', 'font_color': 'white', 'font_name': 'Arial'})
        
        worksheet.set_column('A:A', 35) # Filename
        worksheet.set_column('B:B', 15) # Date
        worksheet.set_column('C:C', 30) # Vendor
        worksheet.set_column('D:D', 12) # Total
        worksheet.set_column('E:E', 15) # Category
        worksheet.set_column('F:F', 50) # Description
        
        # Write Headers based on DYNAMIC list
        for col_num, value in enumerate(summary_cols_to_export):
            worksheet.write(0, col_num, value.capitalize(), header_fmt)
        
        for row_num in range(1, len(df) + 1):
            for col_idx, col_name in enumerate(summary_cols_to_export):
                cell_data = df.iloc[row_num-1][col_name]
                
                if col_name == 'total' and pd.notna(cell_data):
                    worksheet.write(row_num, col_idx, cell_data, currency_fmt)
                elif col_name == 'date':
                    if pd.notna(df.iloc[row_num-1]['date']):
                        worksheet.write_datetime(row_num, col_idx, df.iloc[row_num-1]['date'], date_fmt)
                    else:
                        worksheet.write(row_num, col_idx, cell_data)
                else:
                    if row_num % 2 == 0:
                        cell_fmt = workbook.add_format({'bg_color': '#EBF5FB', 'font_name': 'Arial'})
                    else:
                        cell_fmt = workbook.add_format({'bg_color': 'white', 'font_name': 'Arial'})
                    worksheet.write(row_num, col_idx, str(cell_data), cell_fmt)
        
        worksheet.autofilter(0, 0, len(df), len(summary_cols_to_export) - 1)
        worksheet.freeze_panes(1, 0)

    output.seek(0)
    
    with placeholder_dl.container():
        st.download_button(
            label="📥 Download Excel",
            data=output,
            file_name=f"Receipt_Export_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # 2. METRICS ROW
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Items", len(df))
    with m2:
        st.metric("Total Volume", f"${df['total'].sum():,.2f}")
    with m3:
        avg_trans = df['total'].mean()
        st.metric("Avg. Transaction", f"${avg_trans:,.2f}")

    # 3. CHARTS ROW
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<h3 class="section-header">💵 Spending by Category</h3>', unsafe_allow_html=True)
        st.bar_chart(df.groupby('category')['total'].sum(), use_container_width=False)
    with c2:
        st.markdown('<h3 class="section-header">🧾 Count by Category</h3>', unsafe_allow_html=True)
        st.bar_chart(df['category'].value_counts(), use_container_width=False)

    # 4. DATA TABLE
    st.markdown("<br>", unsafe_allow_html=True)
    st.success("Extraction Complete.")
    
    if 'items' in df.columns:
        st.caption("Note: Line Items exported to 'Line Items' sheet in Excel.")
    
    # Display only columns that exist for the view too
    display_cols = ['display_date', 'vendor', 'total', 'category', 'description']
    display_cols = [c for c in display_cols if c in df.columns]
    
    st.dataframe(df[display_cols], 
                 use_container_width=True, 
                 hide_index=True)

    # 5. ERROR LOG
    if st.session_state.processing_errors:
        with st.expander(f"⚠️ View Errors ({len(st.session_state.processing_errors)})"):
            unique_errors = [dict(t) for t in {tuple(sorted(d.items())) for d in st.session_state.processing_errors}]
            for err in unique_errors:
                st.text(f"📄 {err['file']}: {err['error']}")
