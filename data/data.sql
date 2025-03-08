CREATE TABLE IF NOT EXISTS customerInfo (
    person_customer_id SERIAL PRIMARY KEY,
    person_first_name VARCHAR(50) NOT NULL,
    person_last_name VARCHAR(50) NOT NULL,
    person_email VARCHAR(255) UNIQUE NOT NULL,
    person_phone_number VARCHAR(20),
    person_date_of_birth DATE,
    person_gender CHAR(1),
    person_registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    person_last_login TIMESTAMP,
    person_is_premium BOOLEAN DEFAULT FALSE,
    person_loyalty_points INT DEFAULT 0,
    person_preferred_language VARCHAR(20),
    person_occupation VARCHAR(100),
    person_income NUMERIC(10,2),
    person_marital_status VARCHAR(20),
    
    -- Address Fields
    address_street VARCHAR(255),
    address_city VARCHAR(100),
    address_state VARCHAR(100),
    address_country VARCHAR(100),
    address_postalcode VARCHAR(20),

    -- Account & Payment Info
    account_account_balance NUMERIC(12,2) DEFAULT 0.00,
    account_preferred_payment_method VARCHAR(50),
    account_card_last_four VARCHAR(4),
    account_card_expiry DATE,
    account_has_active_subscription BOOLEAN DEFAULT FALSE,

    -- Shopping Preferences
    preferences_favorite_category VARCHAR(100),
    preferences_avg_spent_per_order NUMERIC(10,2),
    preferences_total_orders INT DEFAULT 0,
    preferences_last_order_date TIMESTAMP,
    preferences_wishlist_items INT DEFAULT 0,
    preferences_newsletter_subscription BOOLEAN DEFAULT TRUE,
    preferences_referral_code VARCHAR(20),

    -- Security & Communication
    securitypassword_hash TEXT NOT NULL,
    security_question VARCHAR(255),
    security_answer_hash TEXT,
    securitytwo_factor_enabled BOOLEAN DEFAULT FALSE,
    securitysms_notifications BOOLEAN DEFAULT TRUE,
    securityemail_notifications BOOLEAN DEFAULT TRUE,
    security_account_status VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id SERIAL PRIMARY KEY,
    campaign_name VARCHAR(255) NOT NULL,
    campaign_type VARCHAR(50) NOT NULL,
    campaign_status VARCHAR(20),
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    budget NUMERIC(12,2),
    actual_spent NUMERIC(12,2) DEFAULT 0.00,
    target_audience VARCHAR(255), -- e.g., 'New Users', 'Returning Customers', 'High Spenders'
    total_reach INT DEFAULT 0, -- Number of people targeted
    impressions INT DEFAULT 0, -- Number of times campaign was viewed
    clicks INT DEFAULT 0, -- Clicks on campaign link
    conversions INT DEFAULT 0, -- Number of successful purchases
    revenue_generated NUMERIC(12,2) DEFAULT 0.00,
    discount_code VARCHAR(50), -- Discount code used in campaign if applicable
    discount_value NUMERIC(10,2), -- Discount value associated with the campaign
    email_open_rate DECIMAL(5,2) , -- Open rate percentage
    email_click_through_rate DECIMAL(5,2) , -- CTR percentage
    cost_per_acquisition NUMERIC(10,2) DEFAULT 0.00, -- CPA metric
    roi DECIMAL(10,2) GENERATED ALWAYS AS ((revenue_generated - actual_spent) / NULLIF(actual_spent, 0) * 100) STORED -- ROI percentage
);

CREATE TABLE IF NOT EXISTS suppliers (
    supplier_unique_identifier SERIAL PRIMARY KEY,
    
    official_supplier_business_name VARCHAR(255) NOT NULL,
    registered_business_address VARCHAR(255),
    primary_contact_person_name VARCHAR(255),
    primary_contact_phone_number VARCHAR(20),
    primary_contact_email_address VARCHAR(255) UNIQUE,
    supplier_country_of_operation VARCHAR(100),
    supplier_tax_identification_number VARCHAR(50),
    preferred_payment_terms_description TEXT,
    total_number_of_products_supplied INT DEFAULT 0,
    average_supplier_rating DECIMAL(3,2) DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS orders (
    order_id SERIAL PRIMARY KEY,
    person_customer_id INT NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    shipping_date TIMESTAMP,
    delivery_date TIMESTAMP,
    order_status VARCHAR(50),
    total_amount NUMERIC(10,2) NOT NULL,
    discount_applied NUMERIC(10,2) DEFAULT 0.00,
    tax_amount NUMERIC(10,2) DEFAULT 0.00,
    shipping_fee NUMERIC(10,2) DEFAULT 0.00,
    payment_status VARCHAR(50),
    payment_method VARCHAR(50),
    tracking_number VARCHAR(50) UNIQUE,
    shipping_address_street VARCHAR(255),
    shipping_address_city VARCHAR(100),
    shipping_address_state VARCHAR(100),
    shipping_address_country VARCHAR(100),
    shipping_address_postalcode VARCHAR(20),
    special_instructions TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Removed ON UPDATE CURRENT_TIMESTAMP (PostgreSQL does not support it)
    campaign_id INT,

    CONSTRAINT fk_orders_customer FOREIGN KEY (person_customer_id) REFERENCES customerInfo(person_customer_id) ON DELETE CASCADE,  -- Added comma
    CONSTRAINT fk_orders_campaign FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id) ON DELETE SET NULL
);


CREATE TABLE IF NOT EXISTS order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    quantity INT,
    price_per_unit NUMERIC(10,2) NOT NULL,
    total_price NUMERIC(10,2) GENERATED ALWAYS AS (quantity * price_per_unit) STORED,
    item_status VARCHAR(50),
    warranty_period INT,
    return_period INT,
    is_returnable BOOLEAN DEFAULT TRUE,
    is_replacement_available BOOLEAN DEFAULT FALSE,
    discount_applied NUMERIC(10,2) DEFAULT 0.00,
    tax_amount NUMERIC(10,2) DEFAULT 0.00,
    shipping_fee NUMERIC(10,2) DEFAULT 0.00,
    item_weight NUMERIC(10,2), -- in kg
    item_dimensions VARCHAR(100), -- e.g., "10x5x2 cm"
    manufacturer VARCHAR(255),

    CONSTRAINT fk_order_items_order FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events (
    event_id SERIAL PRIMARY KEY,
    person_customer_id INT,  -- Links to customerInfo
    order_id INT,            -- Links to orders (if event is order-related)
    order_item_id INT,       -- Links to order_items (if event is item-specific)
    
    event_type VARCHAR(50) NOT NULL,
    
    device_platform VARCHAR(20),
    device_type VARCHAR(50),  -- 'Mobile', 'Desktop', 'Tablet', 'POS Machine' (for offline)
    device_browser VARCHAR(100), -- e.g., 'Chrome', 'Safari', 'Firefox', 'Edge'
    device_os VARCHAR(50),  -- e.g., 'Windows', 'iOS', 'Android'
    device_app_version VARCHAR(20), -- Stores mobile app version if applicable
    device_ip_address VARCHAR(45),  -- Store IPv4 or IPv6
    location_city VARCHAR(100),
    location_country VARCHAR(100),
    session_id VARCHAR(255), -- Unique session tracking for web/app
    referral_source VARCHAR(255), -- Source of visit (Google, Facebook, Direct, etc.)
    utm_campaign VARCHAR(255), -- Marketing campaign tracking
    event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Additional Engagement Metrics
    page_url TEXT, -- URL of the page visited
    time_spent_seconds INT, -- Time spent on action
    click_count INT DEFAULT 0, -- Number of clicks in an interaction
    scroll_depth_percentage INT, -- How far a user scrolled
    
    -- Order & Cart Interaction Data
    cart_value NUMERIC(10,2), -- Total cart value when event occurred
    payment_method VARCHAR(50),
    discount_applied NUMERIC(10,2) DEFAULT 0.00, -- Discount at the time of order

    -- Support & Feedback
    support_ticket_id VARCHAR(50), -- If the event is related to customer support
    review_rating INT, -- User rating for review event
    review_comment TEXT, -- Review content
    return_reason VARCHAR(255), -- Reason for return if applicable
    
    -- Notifications & Promotions
    email_opened BOOLEAN DEFAULT FALSE, -- Whether the user opened an email
    push_notification_clicked BOOLEAN DEFAULT FALSE, -- If a push notification was clicked
    coupon_code_used VARCHAR(50), -- Coupon applied if any
    survey_completed BOOLEAN DEFAULT FALSE, -- If the user completed a survey

    event_metadata TEXT,  -- Optional field for additional JSON-like details

    -- Foreign Keys
    CONSTRAINT fk_events_customer FOREIGN KEY (person_customer_id) REFERENCES customerInfo(person_customer_id) ON DELETE SET NULL,
    CONSTRAINT fk_events_order FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE SET NULL,
    CONSTRAINT fk_events_order_item FOREIGN KEY (order_item_id) REFERENCES order_items(order_item_id) ON DELETE SET NULL
);




CREATE TABLE IF NOT EXISTS products (
    unique_product_identifier SERIAL PRIMARY KEY,
    
    product_display_name VARCHAR(255) NOT NULL,
    detailed_product_description TEXT,
    product_category_primary VARCHAR(100) NOT NULL,
    product_category_secondary VARCHAR(100),
    global_brand_affiliation VARCHAR(100),
    model_identification_code VARCHAR(100),
    stock_keeping_unit_identifier VARCHAR(50) UNIQUE,
    universal_product_code VARCHAR(50) UNIQUE, 
    european_article_number VARCHAR(50) UNIQUE, 
    international_standard_book_number VARCHAR(50) UNIQUE, 
    
    standard_retail_price_including_tax NUMERIC(10,2) NOT NULL,
    promotional_discounted_price NUMERIC(10,2),
    percentage_discount_applied DECIMAL(5,2),
    applicable_value_added_tax DECIMAL(5,2),
    currency_of_transaction VARCHAR(10) DEFAULT 'USD',
    
    available_stock_quantity_in_units INT NOT NULL,
    minimum_threshold_for_restocking INT,
    estimated_replenishment_date DATE,
    associated_supplier_reference_id INT, 
    warehouse_storage_location_details VARCHAR(255),
    production_batch_identifier VARCHAR(100),
    
    net_weight_in_kilograms DECIMAL(10,3),
    physical_length_in_centimeters DECIMAL(10,2),
    physical_width_in_centimeters DECIMAL(10,2),
    physical_height_in_centimeters DECIMAL(10,2),
    volumetric_measurement_in_liters DECIMAL(10,2),
    
    predominant_color_description VARCHAR(50),
    designated_size_variation VARCHAR(50),
    primary_material_composition VARCHAR(100),
    stylistic_representation VARCHAR(100),
    intended_user_demographic VARCHAR(20),
    
    estimated_battery_backup_duration VARCHAR(50),
    energy_consumption_rating VARCHAR(50),
    supported_connectivity_protocols VARCHAR(100),
    embedded_processor_specifications VARCHAR(100),
    integrated_memory_configuration VARCHAR(50),
    total_storage_capacity_details VARCHAR(50),
    
    indexed_search_keywords_for_product TEXT,
    optimized_meta_title_for_seo VARCHAR(255),
    search_engine_meta_description TEXT,
    product_demonstration_video_link TEXT,
    featured_product_flag BOOLEAN DEFAULT FALSE,
    
    aggregate_customer_review_rating DECIMAL(3,2) DEFAULT 0.0,
    total_number_of_verified_reviews INT DEFAULT 0,
    standard_warranty_duration VARCHAR(50),
    comprehensive_return_policy_description TEXT,
    
    shipping_weight_measurement_in_kilograms DECIMAL(10,3),
    fragile_item_indicator BOOLEAN DEFAULT FALSE,
    perishable_product_flag BOOLEAN DEFAULT FALSE,
    expected_lead_time_in_business_days INT,
    
    active_product_status BOOLEAN DEFAULT TRUE,
    official_product_release_date DATE,
    official_product_discontinuation_date DATE,
    
    legal_manufacturer_entity_name VARCHAR(255),
    country_of_product_origin VARCHAR(100),
    estimated_production_cost_per_unit NUMERIC(10,2),
    
    certified_regulatory_compliance_details TEXT,
    environmentally_sustainable_product BOOLEAN DEFAULT FALSE,
    applicable_warranty_coverage_type VARCHAR(50),
    
    CONSTRAINT fk_products_supplier FOREIGN KEY (associated_supplier_reference_id) REFERENCES suppliers(supplier_unique_identifier) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS customers_loyalty_program (
    loyalty_membership_unique_identifier SERIAL PRIMARY KEY,
    
    associated_customer_reference_id INT NOT NULL,
    loyalty_program_tier_level VARCHAR(50),
    accumulated_loyalty_points_balance INT DEFAULT 0,
    last_loyalty_point_update_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    eligible_for_special_promotions BOOLEAN DEFAULT FALSE,
    
    initial_enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_loyalty_tier_upgrade_date TIMESTAMP,
    next_loyalty_tier_evaluation_date TIMESTAMP,
    expiration_date_of_loyalty_points TIMESTAMP,
    
    total_discount_amount_redeemed NUMERIC(10,2) DEFAULT 0.00,
    lifetime_loyalty_points_earned INT DEFAULT 0,
    lifetime_loyalty_points_redeemed INT DEFAULT 0,
    
    exclusive_coupon_codes_assigned TEXT,
    customer_birthday_special_discount BOOLEAN DEFAULT FALSE,
    personalized_product_recommendations JSON,
    
    annual_loyalty_spending_threshold NUMERIC(10,2),
    free_shipping_eligibility BOOLEAN DEFAULT FALSE,
    anniversary_reward_voucher_status BOOLEAN DEFAULT FALSE,
    
    customer_feedback_engagement_score DECIMAL(5,2),
    bonus_loyalty_points_last_month INT DEFAULT 0,
    
    referral_bonus_points_earned INT DEFAULT 0,
    referred_friends_count INT DEFAULT 0,
    
    extra_reward_credits_from_surveys INT DEFAULT 0,
    special_event_invitation_status BOOLEAN DEFAULT FALSE,
    
    redemption_activity_log JSON,
    last_redemption_date TIMESTAMP,
    
    preferred_communication_channel VARCHAR(50),
    participation_in_exclusive_beta_testing BOOLEAN DEFAULT FALSE,
    exclusive_member_early_access BOOLEAN DEFAULT FALSE,
    
    FOREIGN KEY (associated_customer_reference_id) REFERENCES customerInfo(person_customer_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS product_reviews_and_ratings (
    review_unique_identifier SERIAL PRIMARY KEY,
    
    referenced_product_identifier INT NOT NULL,
    reviewing_customer_identifier INT NOT NULL,
    
    customer_review_submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    textual_review_feedback TEXT,
    submitted_review_star_rating DECIMAL(3,2) DEFAULT 0.00,
    
    verification_status_of_reviewer BOOLEAN DEFAULT FALSE,
    number_of_helpful_votes_received INT DEFAULT 0,
    flagged_as_inappropriate BOOLEAN DEFAULT FALSE,
    
    contains_multimedia_content BOOLEAN DEFAULT FALSE,
    associated_review_image_urls TEXT,
    associated_review_video_links TEXT,
    
    sentiment_analysis_score DECIMAL(5,2),
    keywords_extracted_from_review TEXT,
    length_of_review_in_characters INT,
    
    previous_product_purchases_count INT DEFAULT 0,
    return_request_status BOOLEAN DEFAULT FALSE,
    
    response_from_brand_or_seller TEXT,
    response_submission_date TIMESTAMP,
    
    additional_comments_by_other_users JSON,
    user_has_edited_review BOOLEAN DEFAULT FALSE,
    
    total_number_of_edits_made INT DEFAULT 0,
    review_approval_moderation_status VARCHAR(50),
    review_moderator_notes TEXT,
    
    FOREIGN KEY (referenced_product_identifier) REFERENCES products(unique_product_identifier) ON DELETE CASCADE,
    FOREIGN KEY (reviewing_customer_identifier) REFERENCES customerInfo(person_customer_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS transactions_and_payments (
    transaction_unique_identifier SERIAL PRIMARY KEY,
    
    linked_order_reference_identifier INT NOT NULL,
    corresponding_customer_reference_identifier INT NOT NULL,
    
    total_transaction_amount NUMERIC(10,2) NOT NULL,
    applied_discount_value NUMERIC(10,2) DEFAULT 0.00,
    final_billed_amount NUMERIC(10,2) NOT NULL,
    
    transaction_status VARCHAR(50),
    transaction_date_and_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    payment_method_used VARCHAR(50),
    payment_gateway_used VARCHAR(100),
    transaction_authorization_code VARCHAR(100),
    
    billing_address_street VARCHAR(255),
    billing_address_city VARCHAR(100),
    billing_address_state VARCHAR(100),
    billing_address_country VARCHAR(100),
    billing_address_zip_code VARCHAR(20),
    
    shipping_address_street VARCHAR(255),
    shipping_address_city VARCHAR(100),
    shipping_address_state VARCHAR(100),
    shipping_address_country VARCHAR(100),
    shipping_address_zip_code VARCHAR(20),
    
    transaction_currency_code VARCHAR(10) DEFAULT 'USD',
    foreign_exchange_conversion_rate DECIMAL(10,4),
    
    refund_status BOOLEAN DEFAULT FALSE,
    refund_initiation_date TIMESTAMP,
    refund_amount NUMERIC(10,2),
    
    chargeback_request_status BOOLEAN DEFAULT FALSE,
    chargeback_dispute_reason TEXT,
    chargeback_resolution_status VARCHAR(50),
    
    associated_loyalty_points_earned INT DEFAULT 0,
    gift_card_or_store_credit_usage BOOLEAN DEFAULT FALSE,
    applied_gift_card_code VARCHAR(50),
    
    recurring_billing_flag BOOLEAN DEFAULT FALSE,
    installment_payment_status BOOLEAN DEFAULT FALSE,
    
    first_time_customer_transaction BOOLEAN DEFAULT FALSE,
    transaction_frequency_category VARCHAR(50),
    
    digital_wallet_used VARCHAR(50),
    cryptocurrency_payment_flag BOOLEAN DEFAULT FALSE,
    cryptocurrency_type VARCHAR(50),
    
    alternative_payment_method_used VARCHAR(50),
    
    promotional_offer_applied BOOLEAN DEFAULT FALSE,
    special_financing_option_used BOOLEAN DEFAULT FALSE,
    
    customer_feedback_on_transaction TEXT,
    transaction_review_score DECIMAL(3,2),
    
    is_transaction_fraudulent BOOLEAN DEFAULT FALSE,
    fraud_detection_flagged BOOLEAN DEFAULT FALSE,
    fraud_detection_notes TEXT,
    
    FOREIGN KEY (linked_order_reference_identifier) REFERENCES orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY (corresponding_customer_reference_identifier) REFERENCES customerInfo(person_customer_id) ON DELETE SET NULL
);

-- Inventory table to track stock of products
CREATE TABLE IF NOT EXISTS inventory (
    id SERIAL PRIMARY KEY,
    referenced_product_id INT NOT NULL,
    quantity INT NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    warehouse_location VARCHAR(255),
    stock_threshold INT DEFAULT 10,
    safety_stock INT DEFAULT 5,
    supplier_id INT,
    last_restock_date TIMESTAMP,
    expected_restock_date TIMESTAMP,
    purchase_price NUMERIC(10,2),
    bulk_discount NUMERIC(10,2) DEFAULT 0.00,
    storage_temperature VARCHAR(50),
    shelf_life INT,
    batch_number VARCHAR(50),
    expiry_date DATE,
    stock_status VARCHAR(50) DEFAULT 'Available',
    last_inventory_audit_date TIMESTAMP,
    inventory_adjustment_reason TEXT,
    damaged_stock INT DEFAULT 0,
    inbound_shipment_tracking VARCHAR(100),
    outbound_shipment_tracking VARCHAR(100),
    inventory_turnover_rate DECIMAL(10,2) GENERATED ALWAYS AS 
        (quantity / NULLIF(safety_stock, 0)) STORED,
    last_sold_date TIMESTAMP,
    FOREIGN KEY (referenced_product_id) REFERENCES products(unique_product_identifier) ON DELETE CASCADE,
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_unique_identifier) ON DELETE SET NULL
);

-- Shipping table to track order shipments
CREATE TABLE IF NOT EXISTS shipping (
    id SERIAL PRIMARY KEY,
    fk_order_id INT NOT NULL,
    shipping_address TEXT NOT NULL,
    shipping_city VARCHAR(100) NOT NULL,
    shipping_state VARCHAR(100) NOT NULL,
    shipping_zipcode VARCHAR(20) NOT NULL,
    shipping_country VARCHAR(100) NOT NULL,
    shipping_status VARCHAR(50) CHECK (shipping_status IN ('Pending', 'Shipped', 'Delivered', 'Cancelled')),
    tracking_number VARCHAR(255) UNIQUE,
    estimated_delivery DATE,
    shipped_date TIMESTAMP,
    carrier VARCHAR(100),
    shipping_cost DECIMAL(10,2) CHECK (shipping_cost >= 0),
    FOREIGN KEY (fk_order_id) REFERENCES orders(order_id) ON DELETE CASCADE
);

-- Cart table to store user's selected products before checkout
CREATE TABLE IF NOT EXISTS cart (
    id SERIAL PRIMARY KEY,
    fk_user_id INT NOT NULL,
    fk_product_id INT NOT NULL,
    quantity INT NOT NULL DEFAULT 1,
    price_per_unit NUMERIC(10,2) NOT NULL,
    total_price NUMERIC(10,2),  -- Will be updated via trigger
    discount_applied NUMERIC(10,2) DEFAULT 0.00,
    coupon_code VARCHAR(50),
    discounted_total_price NUMERIC(10,2),  -- Will be updated via trigger
    cart_status VARCHAR(20) DEFAULT 'active',
    session_id VARCHAR(255),
    last_activity_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tax_amount NUMERIC(10,2) DEFAULT 0.00,
    shipping_fee NUMERIC(10,2) DEFAULT 0.00,
    estimated_delivery_date DATE,
    is_gift BOOLEAN DEFAULT FALSE,
    gift_message TEXT,
    recommended_products JSON,
    wishlist_flag BOOLEAN DEFAULT FALSE,
    abandonment_reason VARCHAR(255),
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fk_user_id) REFERENCES customerInfo(person_customer_id) ON DELETE CASCADE,
    FOREIGN KEY (fk_product_id) REFERENCES products(unique_product_identifier) ON DELETE CASCADE
);

-- Create ENUM types for PostgreSQL before using them in the table
CREATE TYPE wishlist_status_enum AS ENUM ('active', 'purchased', 'removed');
CREATE TYPE priority_level_enum AS ENUM ('low', 'medium', 'high');
CREATE TYPE added_from_source_enum AS ENUM ('website', 'mobile_app', 'email', 'social_media');

-- Wishlist table to track products a user wants to buy later
CREATE TABLE IF NOT EXISTS wishlist (
    id SERIAL PRIMARY KEY,
    fk_user_id INT NOT NULL,
    fk_product_id INT NOT NULL,
    product_name VARCHAR(255),  -- Stores the product name
    price_at_addition DECIMAL(10,2),  -- Price when added to wishlist
    discount_at_addition DECIMAL(10,2),  -- Discount applied at the time of addition
    wishlist_status wishlist_status_enum DEFAULT 'active',  -- Tracks status of wishlist item
    priority_level priority_level_enum DEFAULT 'medium',  -- Priority of wishlist item
    expected_purchase_date DATE,  -- Expected purchase date if user sets one
    quantity INT DEFAULT 1,  -- Number of units user wants
    notes TEXT,  -- User notes regarding the item
    reminder_set BOOLEAN DEFAULT FALSE,  -- Whether a reminder is set
    reminder_date TIMESTAMP,  -- Reminder date for purchase
    last_viewed_at TIMESTAMP,  -- When user last viewed this item in wishlist
    added_from_source added_from_source_enum,  -- Source of addition
    stock_status_at_addition BOOLEAN DEFAULT TRUE,  -- Whether item was in stock at addition
    category VARCHAR(255),  -- Product category
    brand_name VARCHAR(255),  -- Brand name of the product
    session_id VARCHAR(255),  -- Session ID for tracking user behavior
    currency VARCHAR(10) DEFAULT 'USD',  -- Currency type
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fk_user_id) REFERENCES customerInfo(person_customer_id) ON DELETE CASCADE,
    FOREIGN KEY (fk_product_id) REFERENCES products(unique_product_identifier) ON DELETE CASCADE
);

-- Refunds and Returns table to track return requests
CREATE TABLE IF NOT EXISTS refunds_returns (
    id SERIAL PRIMARY KEY,
    fk_user_id INT NOT NULL,
    fk_order_id INT NOT NULL,
    fk_product_id INT NOT NULL,
    reason TEXT NOT NULL,
    status VARCHAR(50) CHECK (status IN ('Pending', 'Approved', 'Rejected', 'Processed')),
    request_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_date TIMESTAMP,  -- Date when refund/return was processed
    refund_amount DECIMAL(10,2),  -- Amount refunded to the user
    refund_method VARCHAR(50) CHECK (refund_method IN ('Original Payment Method', 'Store Credit', 'Bank Transfer')),  -- Refund method
    return_type VARCHAR(50) CHECK (return_type IN ('Full Return', 'Partial Return', 'Exchange')),  -- Type of return
    tracking_number VARCHAR(255),  -- Tracking number for return shipment
    shipping_carrier VARCHAR(255),  -- Carrier handling return shipping
    restocking_fee DECIMAL(10,2) DEFAULT 0.00,  -- Any restocking fees deducted from refund
    return_condition VARCHAR(50) CHECK (return_condition IN ('New', 'Used', 'Damaged', 'Defective')),  -- Condition of the returned item
    customer_notes TEXT,  -- Notes from the customer about the return
    admin_notes TEXT,  -- Notes from the admin handling the return
    refund_status VARCHAR(50) CHECK (refund_status IN ('Initiated', 'In Progress', 'Completed', 'Failed')),  -- Status of the refund process
    return_label_url TEXT,  -- URL for downloading the return shipping label
    is_refundable BOOLEAN DEFAULT TRUE,  -- Whether the product is eligible for a refund
    refund_initiated_by VARCHAR(50) CHECK (refund_initiated_by IN ('Customer', 'Admin', 'System')),  -- Who initiated the refund
    FOREIGN KEY (fk_user_id) REFERENCES customerInfo(person_customer_id) ON DELETE CASCADE,
    FOREIGN KEY (fk_order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY (fk_product_id) REFERENCES products(unique_product_identifier) ON DELETE CASCADE
);

-- Insert dummy data into customerInfo
INSERT INTO customerInfo (
    person_first_name, person_last_name, person_email, person_phone_number,
    person_date_of_birth, person_gender, person_registration_date, person_last_login,
    person_is_premium, person_loyalty_points, person_preferred_language, person_occupation,
    person_income, person_marital_status, address_street, address_city, address_state,
    address_country, address_postalcode, account_account_balance, 
    account_preferred_payment_method, account_card_last_four, account_card_expiry,
    account_has_active_subscription, preferences_favorite_category,
    preferences_avg_spent_per_order, preferences_total_orders, 
    preferences_last_order_date, preferences_wishlist_items,
    preferences_newsletter_subscription, preferences_referral_code,
    securitypassword_hash, security_question, security_answer_hash,
    securitytwo_factor_enabled, securitysms_notifications,
    securityemail_notifications, security_account_status
) VALUES 
    ('John', 'Doe', 'john.doe@email.com', '123-456-7890', '1990-01-15', 'M', 
    '2023-01-01', '2024-01-15', true, 500, 'English', 'Engineer', 75000.00, 
    'Single', '123 Main St', 'New York', 'NY', 'USA', '10001', 1500.50, 
    'Credit Card', '4321', '2025-12-31', true, 'Electronics', 450.00, 12, 
    '2024-01-10', 5, true, 'REF123', 'hash123', 'First pet name?', 'pethash123',
    true, true, true, 'Active'),

    ('Jane', 'Smith', 'jane.smith@email.com', '234-567-8901', '1985-03-20', 'F',
    '2023-02-15', '2024-01-14', false, 200, 'English', 'Teacher', 60000.00,
    'Married', '456 Oak Ave', 'Los Angeles', 'CA', 'USA', '90001', 750.25,
    'PayPal', '8765', '2025-06-30', false, 'Books', 200.00, 8,
    '2024-01-05', 3, true, 'REF124', 'hash124', 'Mother maiden name?', 'momhash124',
    false, true, true, 'Active'),

    ('Michael', 'Johnson', 'michael.j@email.com', '345-678-9012', '1988-07-10', 'M',
    '2023-03-20', '2024-01-13', true, 750, 'English', 'Doctor', 120000.00,
    'Married', '789 Pine St', 'Chicago', 'IL', 'USA', '60601', 2500.75,
    'Credit Card', '9999', '2026-01-31', true, 'Health', 800.00, 15,
    '2024-01-12', 8, true, 'REF125', 'hash125', 'High school name?', 'schoolhash125',
    true, true, true, 'Active'),

    ('Sarah', 'Williams', 'sarah.w@email.com', '456-789-0123', '1992-05-25', 'F',
    '2023-04-10', '2024-01-15', false, 300, 'Spanish', 'Designer', 65000.00,
    'Single', '321 Elm St', 'Miami', 'FL', 'USA', '33101', 950.00,
    'Debit Card', '5432', '2025-09-30', false, 'Fashion', 350.00, 6,
    '2024-01-08', 12, true, 'REF126', 'hash126', 'Favorite color?', 'colorhash126',
    false, true, true, 'Active'),

    ('David', 'Brown', 'david.b@email.com', '567-890-1234', '1983-11-30', 'M',
    '2023-05-05', '2024-01-14', true, 1000, 'English', 'Lawyer', 95000.00,
    'Divorced', '654 Maple Dr', 'Boston', 'MA', 'USA', '02108', 3500.25,
    'Credit Card', '1111', '2025-08-31', true, 'Books', 550.00, 20,
    '2024-01-11', 4, false, 'REF127', 'hash127', 'First car?', 'carhash127',
    true, false, true, 'Active')
    -- Continue with more rows...
;

-- Insert dummy data into suppliers
INSERT INTO suppliers (
    official_supplier_business_name, registered_business_address,
    primary_contact_person_name, primary_contact_phone_number,
    primary_contact_email_address, supplier_country_of_operation,
    supplier_tax_identification_number, preferred_payment_terms_description,
    total_number_of_products_supplied, average_supplier_rating
) VALUES 
    ('Tech Supplies Inc', '789 Business Ave, Suite 100, San Jose, CA 95110',
    'Robert Wilson', '555-0123', 'robert@techsupplies.com', 'USA',
    'TAX123456789', 'Net 30 days', 150, 4.8),

    ('Global Electronics', '456 Industry Rd, Shenzhen', 'Sarah Chen',
    '555-0124', 'sarah@globalelec.com', 'China', 'GTAX987654321',
    'Net 45 days', 300, 4.5),

    ('Quality Goods Ltd', '123 Commerce St, London', 'James Brown',
    '555-0125', 'james@qualitygoods.com', 'UK', 'UKTAX456789',
    'Net 60 days', 200, 4.2),

    ('Premium Tech Solutions', '321 Tech Park, Seoul', 'Min-Ji Kim',
    '555-0126', 'minji@premiumtech.com', 'South Korea', 'KTAX789012',
    'Net 30 days', 175, 4.7),

    ('Euro Distributors GmbH', '987 Industrial Str, Berlin', 'Hans Mueller',
    '555-0127', 'hans@eurodist.com', 'Germany', 'DETAX345678',
    'Net 45 days', 250, 4.6)
    -- Continue with more rows...
;

-- Insert dummy data into products
INSERT INTO products (
    product_display_name, detailed_product_description, product_category_primary,
    product_category_secondary, global_brand_affiliation, model_identification_code,
    stock_keeping_unit_identifier, universal_product_code, european_article_number,
    international_standard_book_number, standard_retail_price_including_tax,
    promotional_discounted_price, percentage_discount_applied, applicable_value_added_tax,
    currency_of_transaction, available_stock_quantity_in_units, minimum_threshold_for_restocking,
    estimated_replenishment_date, associated_supplier_reference_id, warehouse_storage_location_details,
    production_batch_identifier, net_weight_in_kilograms, physical_length_in_centimeters,
    physical_width_in_centimeters, physical_height_in_centimeters, volumetric_measurement_in_liters,
    predominant_color_description, designated_size_variation, primary_material_composition,
    stylistic_representation, intended_user_demographic, estimated_battery_backup_duration,
    energy_consumption_rating, supported_connectivity_protocols, embedded_processor_specifications,
    integrated_memory_configuration, total_storage_capacity_details, indexed_search_keywords_for_product,
    optimized_meta_title_for_seo, search_engine_meta_description, product_demonstration_video_link,
    featured_product_flag, aggregate_customer_review_rating, total_number_of_verified_reviews,
    standard_warranty_duration, comprehensive_return_policy_description, shipping_weight_measurement_in_kilograms,
    fragile_item_indicator, perishable_product_flag, expected_lead_time_in_business_days,
    active_product_status, official_product_release_date, official_product_discontinuation_date,
    legal_manufacturer_entity_name, country_of_product_origin, estimated_production_cost_per_unit,
    certified_regulatory_compliance_details, environmentally_sustainable_product,
    applicable_warranty_coverage_type
) VALUES 
    ('iPhone 15 Pro', 'Latest flagship smartphone from Apple', 'Electronics', 
    'Smartphones', 'Apple', 'IP15P-2023', 'SKU123456', 'UPC123456789', 
    'EAN123456789', NULL, 999.99, 899.99, 10.00, 20.00, 'USD', 500, 50, 
    '2024-02-15', 1, 'Zone A-123', 'BATCH2024-001', 0.240, 14.7, 7.2, 0.8, 
    0.12, 'Space Gray', 'Standard', 'Aluminum and Glass', 'Modern', 'Premium', 
    '24 hours', 'A+++', 'WiFi 6E, 5G, Bluetooth 5.3', 'A17 Pro Chip', 
    '8GB RAM', '256GB', 'iphone, smartphone, apple, pro, camera', 
    'iPhone 15 Pro - Latest Premium Smartphone', 'Experience the next generation iPhone with revolutionary features',
    'https://youtube.com/iphone15pro', true, 4.8, 1250, '12 months', 
    '14-day return policy with original packaging', 0.390, true, false, 3, 
    true, '2023-09-22', NULL, 'Apple Inc.', 'China', 400.00,
    'FCC, CE, RoHS compliant', true, 'Limited Warranty'),

    ('Samsung 4K Smart TV', 'Premium QLED 4K Smart TV', 'Electronics',
    'Television', 'Samsung', 'QN65Q80C', 'SKU789012', 'UPC987654321',
    'EAN987654321', NULL, 1499.99, 1299.99, 13.33, 20.00, 'USD', 200, 20,
    '2024-02-20', 2, 'Zone B-456', 'BATCH2024-002', 25.500, 145.0, 85.0, 
    7.0, NULL, 'Black', '65-inch', 'Metal and Glass', 'Contemporary', 'All',
    NULL, 'A+', 'WiFi, Bluetooth, HDMI 2.1', 'Neural Quantum Processor 4K',
    NULL, NULL, 'tv, samsung, 4k, smart tv, qled', 'Samsung 65" QLED 4K Smart TV',
    'Experience stunning 4K picture quality with AI-powered processing',
    'https://youtube.com/samsungTV', true, 4.7, 850, '24 months',
    '30-day return policy', 28.000, true, false, 5, true, '2023-06-15',
    NULL, 'Samsung Electronics', 'South Korea', 800.00, 'Energy Star, CE',
    true, 'Extended Warranty Available')
    -- Continue with more products...
;

-- Insert dummy data into campaigns
INSERT INTO campaigns (
    campaign_name, campaign_type, campaign_status, start_date, end_date,
    budget, actual_spent, target_audience, total_reach, impressions,
    clicks, conversions, revenue_generated, discount_code, discount_value,
    email_open_rate, email_click_through_rate, cost_per_acquisition
) VALUES 
    ('Summer Sale 2024', 'Seasonal', 'Planned', '2024-06-01', '2024-06-30',
    50000.00, 0.00, 'All Customers', 100000, 250000, 25000, 2500,
    125000.00, 'SUMMER24', 20.00, 25.50, 12.75, 20.00),

    ('New Customer Welcome', 'Promotional', 'Active', '2024-01-01', '2024-12-31',
    25000.00, 15000.00, 'New Users', 50000, 120000, 12000, 1800,
    90000.00, 'WELCOME2024', 15.00, 35.20, 18.40, 8.33),

    ('Black Friday 2024', 'Flash Sale', 'Planned', '2024-11-29', '2024-11-30',
    100000.00, 0.00, 'All Customers', 200000, 500000, 50000, 5000,
    250000.00, 'BF2024', 30.00, 45.00, 22.50, 20.00)
    -- Continue with more campaigns...
;

-- Insert dummy data into orders
INSERT INTO orders (
    person_customer_id, order_date, shipping_date, delivery_date,
    order_status, total_amount, discount_applied, tax_amount,
    shipping_fee, payment_status, payment_method, tracking_number,
    shipping_address_street, shipping_address_city, shipping_address_state,
    shipping_address_country, shipping_address_postalcode,
    special_instructions, campaign_id
) VALUES 
    (1, '2024-01-15 10:30:00', '2024-01-16 14:20:00', '2024-01-18 15:00:00',
    'Delivered', 1299.99, 50.00, 104.00, 15.00, 'Paid', 'Credit Card',
    'TRACK123ABC', '123 Main St', 'New York', 'NY', 'USA', '10001',
    'Please leave at front door', 1),

    (2, '2024-01-16 15:45:00', '2024-01-17 09:30:00', NULL,
    'Shipped', 799.99, 25.00, 64.00, 12.00, 'Paid', 'PayPal',
    'TRACK456DEF', '456 Oak Ave', 'Los Angeles', 'CA', 'USA', '90001',
    'Signature required', 2),

    (3, '2024-01-17 08:20:00', NULL, NULL, 'Processing',
    2499.99, 100.00, 200.00, 0.00, 'Paid', 'Credit Card',
    'TRACK789GHI', '789 Pine St', 'Chicago', 'IL', 'USA', '60601',
    NULL, 1)
    -- Continue with more orders...
;

-- Insert dummy data into order_items
INSERT INTO order_items (
    order_id, product_id, product_name, quantity, price_per_unit,
    item_status, warranty_period, return_period, is_returnable,
    is_replacement_available, discount_applied, tax_amount,
    shipping_fee, item_weight, item_dimensions, manufacturer
) VALUES 
    (1, 1, 'iPhone 15 Pro', 1, 999.99, 'Delivered', 12, 14, true,
    true, 50.00, 80.00, 15.00, 0.240, '14.7x7.2x0.8', 'Apple Inc.'),

    (2, 2, 'Samsung 4K Smart TV', 1, 1499.99, 'Shipped', 24, 30, true,
    true, 100.00, 120.00, 50.00, 25.500, '145x85x7', 'Samsung Electronics'),

    (3, 1, 'iPhone 15 Pro', 2, 999.99, 'Processing', 12, 14, true,
    true, 100.00, 160.00, 0.00, 0.480, '14.7x7.2x0.8', 'Apple Inc.')
    -- Continue with more order items...
;

-- Insert dummy data into events
INSERT INTO events (
    person_customer_id, order_id, order_item_id, event_type,
    device_platform, device_type, device_browser, device_os,
    device_app_version, device_ip_address, location_city,
    location_country, session_id, referral_source, utm_campaign,
    page_url, time_spent_seconds, click_count, scroll_depth_percentage,
    cart_value, payment_method, discount_applied, support_ticket_id,
    review_rating, review_comment, return_reason, email_opened,
    push_notification_clicked, coupon_code_used, survey_completed,
    event_metadata
) VALUES 
    (1, 1, 1, 'Purchase', 'Web', 'Desktop', 'Chrome', 'Windows',
    NULL, '192.168.1.1', 'New York', 'USA', 'sess_123abc',
    'Google Search', 'winter_sale', 'https://example.com/checkout',
    300, 15, 85, 999.99, 'Credit Card', 50.00, NULL,
    NULL, NULL, NULL, true, false, 'WINTER10', false,
    '{"browser_version": "97.0.4692", "screen_resolution": "1920x1080"}'),

    (2, 2, 2, 'Cart Add', 'Mobile', 'Smartphone', 'Safari', 'iOS',
    '2.1.0', '192.168.1.2', 'Los Angeles', 'USA', 'sess_456def',
    'Direct', NULL, 'https://example.com/product/123',
    120, 5, 65, 1499.99, NULL, NULL, NULL,
    NULL, NULL, NULL, false, true, NULL, false,
    '{"app_version": "2.1.0", "device_model": "iPhone 13"}'),

    (3, 3, 3, 'Review', 'App', 'Tablet', 'Chrome', 'Android',
    '2.0.1', '192.168.1.3', 'Chicago', 'USA', 'sess_789ghi',
    'Email', 'feedback_campaign', 'https://example.com/review',
    180, 8, 100, NULL, NULL, NULL, NULL,
    5, 'Great product!', NULL, true, true, NULL, true,
    '{"rating_categories": {"quality": 5, "value": 4, "delivery": 5}}')
    -- Continue with more events...
;

-- Insert dummy data into customers_loyalty_program
INSERT INTO customers_loyalty_program (
    associated_customer_reference_id, loyalty_program_tier_level,
    accumulated_loyalty_points_balance, eligible_for_special_promotions,
    initial_enrollment_date, last_loyalty_tier_upgrade_date,
    next_loyalty_tier_evaluation_date, expiration_date_of_loyalty_points,
    total_discount_amount_redeemed, lifetime_loyalty_points_earned,
    lifetime_loyalty_points_redeemed, exclusive_coupon_codes_assigned,
    customer_birthday_special_discount, personalized_product_recommendations,
    annual_loyalty_spending_threshold, free_shipping_eligibility,
    anniversary_reward_voucher_status, customer_feedback_engagement_score,
    bonus_loyalty_points_last_month, referral_bonus_points_earned,
    referred_friends_count, extra_reward_credits_from_surveys,
    special_event_invitation_status, redemption_activity_log,
    last_redemption_date, preferred_communication_channel,
    participation_in_exclusive_beta_testing, exclusive_member_early_access
) VALUES 
    (1, 'Gold', 5000, true, '2023-01-01', '2023-06-15',
    '2024-06-15', '2024-12-31', 250.00, 7500, 2500,
    'GOLD10,GOLD20', true, 
    '{"recommended": ["product1", "product2", "product3"]}',
    5000.00, true, true, 4.8, 100, 200, 3, 50,
    true, '{"last_redemption": "2024-01-15", "points_used": 500}',
    '2024-01-15', 'Email', true, true),

    (2, 'Silver', 2500, true, '2023-02-01', '2023-08-15',
    '2024-08-15', '2024-12-31', 150.00, 4000, 1500,
    'SILVER10', true,
    '{"recommended": ["product4", "product5"]}',
    2500.00, true, false, 4.2, 50, 100, 1, 25,
    false, '{"last_redemption": "2024-01-10", "points_used": 300}',
    '2024-01-10', 'SMS', false, false)
    -- Continue with more loyalty program entries...
;

-- Insert dummy data into product_reviews_and_ratings
INSERT INTO product_reviews_and_ratings (
    referenced_product_identifier, reviewing_customer_identifier,
    textual_review_feedback, submitted_review_star_rating,
    verification_status_of_reviewer, number_of_helpful_votes_received,
    flagged_as_inappropriate, contains_multimedia_content,
    associated_review_image_urls, associated_review_video_links,
    sentiment_analysis_score, keywords_extracted_from_review,
    length_of_review_in_characters, previous_product_purchases_count,
    return_request_status, response_from_brand_or_seller,
    response_submission_date, additional_comments_by_other_users,
    user_has_edited_review, total_number_of_edits_made,
    review_approval_moderation_status, review_moderator_notes
) VALUES 
    (1, 1, 'Excellent product, exceeded my expectations!', 5.00,
    true, 25, false, true, 'https://example.com/img1,https://example.com/img2',
    'https://example.com/video1', 0.95,
    'excellent,quality,value,recommended', 150, 3, false,
    'Thank you for your positive feedback!', '2024-01-16',
    '{"comments": [{"user": "user1", "text": "Helpful review!"}]}',
    false, 0, 'Approved', 'Verified purchase, compliant review'),

    (2, 2, 'Good product but delivery was delayed', 4.00,
    true, 15, false, false, NULL, NULL, 0.75,
    'good,delivery,delayed', 100, 2, false,
    'We apologize for the delay.', '2024-01-17',
    '{"comments": []}', true, 1, 'Approved',
    'Updated review after seller response')
    -- Continue with more reviews...
;

-- Insert dummy data into transactions_and_payments
INSERT INTO transactions_and_payments (
    linked_order_reference_identifier,
    corresponding_customer_reference_identifier,
    total_transaction_amount, applied_discount_value,
    final_billed_amount, transaction_status,
    payment_method_used, payment_gateway_used,
    transaction_authorization_code, billing_address_street,
    billing_address_city, billing_address_state,
    billing_address_country, billing_address_zip_code,
    shipping_address_street, shipping_address_city,
    shipping_address_state, shipping_address_country,
    shipping_address_zip_code, transaction_currency_code,
    foreign_exchange_conversion_rate, refund_status,
    refund_initiation_date, refund_amount,
    chargeback_request_status, chargeback_dispute_reason,
    chargeback_resolution_status, associated_loyalty_points_earned,
    gift_card_or_store_credit_usage, applied_gift_card_code,
    recurring_billing_flag, installment_payment_status,
    first_time_customer_transaction, transaction_frequency_category,
    digital_wallet_used, cryptocurrency_payment_flag,
    cryptocurrency_type, alternative_payment_method_used,
    promotional_offer_applied, special_financing_option_used,
    customer_feedback_on_transaction, transaction_review_score,
    is_transaction_fraudulent, fraud_detection_flagged,
    fraud_detection_notes
) VALUES 
    (1, 1, 1299.99, 50.00, 1249.99, 'Completed',
    'Credit Card', 'Stripe', 'AUTH123XYZ',
    '123 Main St', 'New York', 'NY', 'USA', '10001',
    '123 Main St', 'New York', 'NY', 'USA', '10001',
    'USD', 1.00, false, NULL, NULL, false, NULL, NULL,
    50, false, NULL, false, false, true, 'First Purchase',
    NULL, false, NULL, NULL, true, false,
    'Smooth transaction', 5.00, false, false, NULL),

    (2, 2, 799.99, 25.00, 774.99, 'Completed',
    'PayPal', 'PayPal', 'PP456ABC',
    '456 Oak Ave', 'Los Angeles', 'CA', 'USA', '90001',
    '456 Oak Ave', 'Los Angeles', 'CA', 'USA', '90001',
    'USD', 1.00, false, NULL, NULL, false, NULL, NULL,
    30, false, NULL, false, false, false, 'Regular',
    'PayPal', false, NULL, NULL, false, false,
    'Quick and easy', 4.50, false, false, NULL)
    -- Continue with more transactions...
;

-- Insert dummy data into inventory
INSERT INTO inventory (
    referenced_product_id, quantity, warehouse_location,
    stock_threshold, safety_stock, supplier_id,
    last_restock_date, expected_restock_date, purchase_price,
    bulk_discount, storage_temperature, shelf_life,
    batch_number, expiry_date, stock_status,
    last_inventory_audit_date, inventory_adjustment_reason,
    damaged_stock, inbound_shipment_tracking,
    outbound_shipment_tracking, last_sold_date
) VALUES 
    (1, 500, 'Warehouse A-123', 50, 25, 1,
    '2024-01-01', '2024-02-01', 800.00, 50.00,
    'Room Temperature', 730, 'BATCH2024A', '2026-01-01',
    'Available', '2024-01-15', NULL, 0,
    'INSHIP123', 'OUTSHIP456', '2024-01-16'),

    (2, 200, 'Warehouse B-456', 20, 10, 2,
    '2024-01-05', '2024-02-05', 1200.00, 100.00,
    'Room Temperature', 365, 'BATCH2024B', '2025-01-05',
    'Available', '2024-01-15', NULL, 2,
    'INSHIP789', 'OUTSHIP012', '2024-01-17')
    -- Continue with more inventory entries...
;

-- Insert dummy data into cart
INSERT INTO cart (
    fk_user_id, fk_product_id, quantity, price_per_unit,
    total_price, discount_applied, coupon_code,
    discounted_total_price, cart_status, session_id,
    tax_amount, shipping_fee, estimated_delivery_date,
    is_gift, gift_message, recommended_products,
    wishlist_flag, abandonment_reason
) VALUES 
    (1, 1, 2, 999.99, 1999.98, 200.00, 'SAVE200',
    1799.98, 'active', 'sess_abc123', 144.00, 15.00,
    '2024-01-20', true, 'Happy Birthday!',
    '{"rec1": "product2", "rec2": "product3"}',
    false, NULL),

    (2, 2, 1, 1499.99, 1499.99, 150.00, 'SAVE150',
    1349.99, 'active', 'sess_def456', 108.00, 25.00,
    '2024-01-21', false, NULL,
    '{"rec1": "product1", "rec2": "product4"}',
    true, NULL)
    -- Continue with more cart entries...
;

-- Insert dummy data into wishlist
INSERT INTO wishlist (
    fk_user_id, fk_product_id, product_name, price_at_addition,
    discount_at_addition, wishlist_status, priority_level,
    expected_purchase_date, quantity, notes, reminder_set,
    reminder_date, last_viewed_at, added_from_source,
    stock_status_at_addition, category, brand_name,
    session_id, currency
) VALUES 
    (1, 1, 'iPhone 15 Pro', 999.99, 0.00, 'active',
    'high', '2024-02-01', 1, 'Wait for sale',
    true, '2024-01-25', '2024-01-16', 'website',
    true, 'Electronics', 'Apple',
    'sess_abc123', 'USD'),

    (2, 2, 'Samsung 4K Smart TV', 1499.99, 150.00,
    'active', 'medium', '2024-03-01', 1,
    'Compare with other models', false, NULL,
    '2024-01-17', 'mobile_app', true, 'Electronics',
    'Samsung', 'sess_def456', 'USD')
    -- Continue with more wishlist entries...
;

-- Insert dummy data into refunds_returns
INSERT INTO refunds_returns (
    fk_user_id, fk_order_id, fk_product_id, reason,
    status, request_date, processed_date, refund_amount,
    refund_method, return_type, tracking_number,
    shipping_carrier, restocking_fee, return_condition,
    customer_notes, admin_notes, refund_status,
    return_label_url, is_refundable, refund_initiated_by
) VALUES 
    (1, 1, 1, 'Wrong size ordered',
    'Approved', '2024-01-15', '2024-01-16', 999.99,
    'Original Payment Method', 'Full Return',
    'RET123ABC', 'FedEx', 0.00, 'New',
    'Item unopened', 'Approved - new condition',
    'Completed', 'https://example.com/return-label/123',
    true, 'Customer'),

    (2, 2, 2, 'Defective product',
    'Pending', '2024-01-16', NULL, 1499.99,
    'Store Credit', 'Exchange', 'RET456DEF',
    'UPS', 0.00, 'Defective',
    'Product not working', 'Pending inspection',
    'In Progress', 'https://example.com/return-label/456',
    true, 'Customer')
    -- Continue with more refunds/returns...
;





