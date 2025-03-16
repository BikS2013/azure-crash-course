schema = [
        {"name": "id", "type": "Edm.String", "key": True, "filterable": True},
        {"name": "content", "type": "Edm.String", "filterable": False, "sortable": False, "searchable": True, "analyzer": "standard.lucene"},
        {"name": "chunk_id", "type": "Edm.String", "filterable": True, "sortable": True, "searchable": False},
        {"name": "type", "type": "Edm.String", "filterable": True, "sortable": True, "searchable": True},
        {"name": "title", "type": "Edm.String", "filterable": True, "sortable": True, "searchable": True}
    ]