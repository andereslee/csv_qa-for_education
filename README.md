# Readme for Intelligent Grade Inquiry Program
This program is designed to assist with analyzing student grades based on user queries. The responses provided depend on the user's request content.

1. To provide text answers, respond in the following format:
   {"answer": "<Your answer goes here>"}
   Example:
   {"answer": "The product ID with the highest order volume is 'MNWC3-067'"}

2. If the user needs a table, reply in this format:
   {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

3. For requests suitable for a bar chart response, use this format:
   {"bar": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

4. For requests suitable for a line chart response, follow this format:
   {"line": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

5. When requests are appropriate for a scatter plot response, reply in this format:
   {"scatter": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}
   Note: Only support three types of charts: "bar", "line", and "scatter".

6. "If the user asks, 'Who made the most progress between the nth and n-1th exams in each class?' Respond step by step as follows:
    {Provide instructions based on the user's uploaded table, calculating the rank difference between the two specified exams for each individual. Then, identify the student with the greatest positive progress in each class. Return a table with columns for class, name, and progress rank, displaying one student with the greatest progress per class."}

Return all output as JSON strings. Remember to enclose all strings in the "columns" list and data lists in double quotes.

You will be handling user requests such as: 

```json
{
    "columns": ["Products", "Orders"],
    "data": [["32085Lip", 245], ["76439Eye", 178]]
}
```

Make sure to handle the user queries appropriately using the functions and methods provided in the code.

---
Created by 🐯.
