def union_tabels(list_tables):
    tables = [f"Select  'index' as 'index' ,name as name ,type as type ,CAST(count as INT) as 'count',count_measure as count_measure,source_id as source_id,CAST(price_current as INT) as price_current,price_current_currency as price_current_currency,CAST(price_old as INT) as price_old, price_old_currency as price_old_currency, url_detail as url_detail, source as source, load_date as load_date, category as category, webfilter as webfilter from {x}" for x in list_tables]
    return ("""
    union
    
    """.join(tables))