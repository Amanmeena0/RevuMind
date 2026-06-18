# RevuMind documentation!

## Description

The Multimodal Product Review Intelligence System is an intelligent platform that analyzes product reviews using both textual and visual data. By leveraging advanced machine learning and natural language processing techniques, the system provides deep insights into product perceptions and customer feedback across multiple modalities.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://revumind-assets/data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://revumind-assets/data/` to `data/`.


