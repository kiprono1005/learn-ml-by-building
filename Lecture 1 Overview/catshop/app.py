import argparse, json, logging, random
import re
from pathlib import Path
from ast import literal_eval

from flask import (
    Flask,
    request,
    redirect,
    url_for,
    jsonify  # Added for JSON responses
)

from rich import print

from catshop.engine.engine import (
    load_products,
    init_search_engine,
    convert_web_app_string_to_var,
    get_top_n_product_from_keywords,
    get_product_per_page,
    map_action_to_html,
    END_BUTTON
)
from catshop.engine.goal import get_reward, get_goals
from catshop.utils import (
    generate_mturk_code,
    setup_logger,
    DEFAULT_FILE_PATH,
    DEBUG_PROD_SIZE,
)
from catshop.cat_classifier import get_cat_classifier, CAT_CATEGORIES

app = Flask(__name__)

search_engine = None
all_products = None
product_item_dict = None
product_prices = None
attribute_to_asins = None
goals = None
weights = None

# Make human/synthetic goals sound cat-like for the UI
def catify_instruction(text: str) -> str:
    if not text:
        return "Purr-quest: find something delightful for a distinguished feline."
    s = text
    # Replace leading 'Find me' (case-insensitive)
    s = re.sub(r'(?i)^find me', 'Find for this very picky cat', s)
    # Only add comma before 'and' if not already preceded by a comma
    # e.g., '..., and ...' stays the same; '... and ...' -> '..., and ...'
    s = re.sub(r'(?<![,]) and ', ', and ', s)
    return f"Purr-quest: {s} 😺"

user_sessions = dict()
user_log_dir = None
SHOW_ATTRS_TAB = False

@app.route('/')
def home():
    return redirect(url_for('index', session_id="abc"))

@app.route('/<session_id>', methods=['GET', 'POST'])
def index(session_id):
    global user_log_dir
    global all_products, product_item_dict, \
           product_prices, attribute_to_asins, \
           search_engine, \
           goals, weights, user_sessions

    if search_engine is None:
        all_products, product_item_dict, product_prices, attribute_to_asins = \
            load_products(
                filepath=DEFAULT_FILE_PATH,
                num_products=DEBUG_PROD_SIZE,
                human_goals=False  # Bypass items_human_ins.json dependency
            )
        search_engine = init_search_engine(num_products=DEBUG_PROD_SIZE)
        goals = get_goals(all_products, product_prices, human_goals=False)
        random.seed(233)
        random.shuffle(goals)
        weights = [goal['weight'] for goal in goals]

    if session_id not in user_sessions and 'fixed' in session_id:
        goal_dix = int(session_id.split('_')[-1])
        goal = goals[goal_dix]
        instruction_text = catify_instruction(goal['instruction_text'])
        user_sessions[session_id] = {'goal': goal, 'done': False}
        if user_log_dir is not None:
            setup_logger(session_id, user_log_dir)
    elif session_id not in user_sessions:
        goal = random.choices(goals, weights)[0]
        instruction_text = catify_instruction(goal['instruction_text'])
        user_sessions[session_id] = {'goal': goal, 'done': False}
        if user_log_dir is not None:
            setup_logger(session_id, user_log_dir)
    else:
        instruction_text = catify_instruction(user_sessions[session_id]['goal']['instruction_text'])

    if request.method == 'POST' and 'search_query' in request.form:
        keywords = request.form['search_query'].lower().split(' ')
        return redirect(url_for(
            'search_results',
            session_id=session_id,
            keywords=keywords,
            page=1,
        ))
    if user_log_dir is not None:
        logger = logging.getLogger(session_id)
        logger.info(json.dumps(dict(
            page='index',
            url=request.url,
            goal=user_sessions[session_id]['goal'],
        )))
    return map_action_to_html(
        'start',
        session_id=session_id,
        instruction_text=instruction_text,
    )


@app.route('/cat_debug', methods=['GET'])
def cat_debug():
    clf = get_cat_classifier()
    info = {
        'using_lora': getattr(clf, 'using_lora', None),
        'model_path': str(getattr(clf, 'model_path', '')),
        'usage_counts': getattr(clf, 'usage_counts', {}),
        'category_counts': getattr(clf, 'category_counts', {}),
    }
    return jsonify(info)

@app.route(
    '/search_results/<session_id>/<keywords>/<page>',
    methods=['GET', 'POST']
)
def search_results(session_id, keywords, page):
    instruction_text = user_sessions[session_id]['goal']['instruction_text']
    page = convert_web_app_string_to_var('page', page)
    keywords = convert_web_app_string_to_var('keywords', keywords)
    top_n_products = get_top_n_product_from_keywords(
        keywords,
        search_engine,
        all_products,
        product_item_dict,
        attribute_to_asins,
    )
    products = get_product_per_page(top_n_products, page)
    # Add cat classifications to products
    cat_classifier = get_cat_classifier()
    for product in products:
        classification = cat_classifier.classify_product(
            product.get('Title', ''),
            product.get('Description', '')
        )
        product['cat_classification'] = classification
    # Debug: print classifier status
    try:
        print(
            "CAT DEBUG | using_lora=",
            getattr(cat_classifier, 'using_lora', None),
            " model_path=",
            getattr(cat_classifier, 'model_path', None),
            " usage_counts=",
            getattr(cat_classifier, 'usage_counts', None),
            " category_counts=",
            getattr(cat_classifier, 'category_counts', None),
        )
    except Exception:
        pass
    html = map_action_to_html(
        'search',
        session_id=session_id,
        products=products,
        keywords=keywords,
        page=page,
        total=len(top_n_products),
        instruction_text=instruction_text,
    )
    logger = logging.getLogger(session_id)
    logger.info(json.dumps(dict(
        page='search_results',
        url=request.url,
        goal=user_sessions[session_id]['goal'],
        content=dict(
            keywords=keywords,
            search_result_asins=[p['asin'] for p in products],
            page=page,
        )
    )))
    return html


@app.route(
    '/item_page/<session_id>/<asin>/<keywords>/<page>/<options>',
    methods=['GET', 'POST']
)
def item_page(session_id, asin, keywords, page, options):
    options = literal_eval(options)
    product_info = product_item_dict[asin]

    goal_instruction = user_sessions[session_id]['goal']['instruction_text']
    product_info['goal_instruction'] = goal_instruction

    html = map_action_to_html(
        'click',
        session_id=session_id,
        product_info=product_info,
        keywords=keywords,
        page=page,
        asin=asin,
        options=options,
        instruction_text=goal_instruction,
        show_attrs=SHOW_ATTRS_TAB,
    )
    logger = logging.getLogger(session_id)
    logger.info(json.dumps(dict(
        page='item_page',
        url=request.url,
        goal=user_sessions[session_id]['goal'],
        content=dict(
            keywords=keywords,
            page=page,
            asin=asin,
            options=options,
        )
    )))
    return html


@app.route(
    '/item_sub_page/<session_id>/<asin>/<keywords>/<page>/<sub_page>/<options>',
    methods=['GET', 'POST']
)
def item_sub_page(session_id, asin, keywords, page, sub_page, options):
    options = literal_eval(options)
    product_info = product_item_dict[asin]

    goal_instruction = user_sessions[session_id]['goal']['instruction_text']
    product_info['goal_instruction'] = goal_instruction

    html = map_action_to_html(
        f'click[{sub_page}]',
        session_id=session_id,
        product_info=product_info,
        keywords=keywords,
        page=page,
        asin=asin,
        options=options,
        instruction_text=goal_instruction
    )
    logger = logging.getLogger(session_id)
    logger.info(json.dumps(dict(
        page='item_sub_page',
        url=request.url,
        goal=user_sessions[session_id]['goal'],
        content=dict(
            keywords=keywords,
            page=page,
            asin=asin,
            options=options,
        )
    )))
    return html


@app.route('/done/<session_id>/<asin>/<options>', methods=['GET', 'POST'])
def done(session_id, asin, options):
    options = literal_eval(options)
    goal = user_sessions[session_id]['goal']
    purchased_product = product_item_dict[asin]
    price = product_prices[asin]

    reward, reward_info = get_reward(
        purchased_product,
        goal,
        price=price,
        options=options,
        verbose=True
    )
    user_sessions[session_id]['done'] = True
    user_sessions[session_id]['reward'] = reward
    print(user_sessions)

    logger = logging.getLogger(session_id)
    logger.info(json.dumps(dict(
        page='done',
        url=request.url,
        goal=goal,
        content=dict(
            asin=asin,
            options=options,
            price=price,
        ),
        reward=reward,
        reward_info=reward_info,
    )))
    del logging.root.manager.loggerDict[session_id]
    
    return map_action_to_html(
        f'click[{END_BUTTON}]',
        session_id=session_id,
        reward=reward,
        asin=asin,
        options=options,
        reward_info=reward_info,
        query=purchased_product['query'],
        category=purchased_product['category'],
        product_category=purchased_product['product_category'],
        goal_attrs=user_sessions[session_id]['goal']['attributes'],
        purchased_attrs=purchased_product['Attributes'],
        goal=goal,
        mturk_code=generate_mturk_code(session_id),
    )

@app.route('/cat_chat/<session_id>', methods=['POST'])
def cat_chat(session_id):
    """Handle cat chat requests"""
    data = request.get_json()
    product_name = data.get('product_name', '')
    question = data.get('question', '')

    # Get cat classifier
    cat_classifier = get_cat_classifier()

    # Generate cat response
    response = cat_classifier.chat_about_product(product_name, question)

    # Log the interaction if logging is enabled
    if user_log_dir is not None:
        logger = logging.getLogger(session_id)
        logger.info(json.dumps(dict(
            page='cat_chat',
            product=product_name,
            question=question,
            response=response
        )))

    return json.dumps({'response': response})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebShop flask app backend configuration")
    parser.add_argument("--log", action='store_true', help="Log actions on WebShop in trajectory file")
    parser.add_argument("--attrs", action='store_true', help="Show attributes tab in item page")
    parser.add_argument("--port", type=int, default=3000, help="Port to run the Flask app on (default: 3000)")

    args = parser.parse_args()
    if args.log:
        user_log_dir = Path('user_session_logs/mturk')
        user_log_dir.mkdir(parents=True, exist_ok=True)
    SHOW_ATTRS_TAB = args.attrs

    app.run(host='0.0.0.0', port=args.port)
