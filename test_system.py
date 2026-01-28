def test_login_success(client):
    token = client.login()
    assert token is not None


def test_all_menu_apis_accessible(client):
    token = client.login()
    menus = client.get_menus(token)

    for m in menus:
        resp = client.call_api(m["api"], token)
        assert resp.status_code == 200, f"API failed: {m['api']}"
