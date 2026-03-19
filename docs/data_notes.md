# Data Notes

## loan_status
- Values: `Fully Paid`, `Charged Off`, `Late (31-120 days)`, `Late (16-30 days)`, `Current`, `In Grace Period`
- Mappings:
    - Default = 1: `Charged Off`, `Late (31-120 days)`, `Late (16-30 days)`
    - Default = 0: `Fully Paid`
    - Drop entirely: `Current`, `In Grace Period`
- Logic: `Current` and `In Grace Period` filter >93% of the data with unknown outcome or too volatile of an amibiguity for the data science. 

## missing_values -> post_default_filter
- months_since_* -> 9%-78%
    - Logic: Reliant on missed payments which have small probability in `Fully Paid` loan_status, making  ~80% of list.
- emp_length, emp_title -> 6%