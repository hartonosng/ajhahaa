WITH product_json AS (
    SELECT
        goal_id,
        product_type,
        CONCAT('{"accountid":"', accountid, '","current_balance":', current_balance, '}') AS account_json
    FROM accounts
),

json_array AS (
    SELECT
        goal_id,
        product_type,
        CONCAT('[', CONCAT_WS(',', collect_list(account_json)), ']') AS accounts_json
    FROM product_json
    GROUP BY goal_id, product_type
),

json_object AS (
    SELECT
        goal_id,
        CONCAT('\"', product_type, '\":', accounts_json) AS product_json
    FROM json_array
),

final_json AS (
    SELECT
        goal_id,
        CONCAT('{', CONCAT_WS(',', collect_list(product_json)), '}') AS account_json
    FROM json_object
    GROUP BY goal_id
)

SELECT goal_id, account_json FROM final_json;
