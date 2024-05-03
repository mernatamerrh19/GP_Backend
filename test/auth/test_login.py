from doctors_app.models import Doctor


def test_create_user(client, db):
    rp = client.post(
        "/doctors/doctors/",
        format="json",
        data={
            "last_name": "01064160586",
            "email": "om23440@gmail.com",
            "first_name": "FCMToken",
            "password": "123456789",

            # "image": ("wwww-removebg-preview.png", open("C:/Users/DELL/Pictures/wwww-removebg-preview.png", "rb"))
        },
    )
    print(rp.data)
    assert rp.status_code == 201, rp.data
    assert "password" not in str(rp.data)
    # assert "status" in rp.data and "message" in rp.data
    rp = client.post(
        "/doctors/patient/",
        format="json",
        data={
            "last_name": "01064160586",
            "email": "om23440+5@gmail.com",
            "first_name": "FCMToken",
            "password": "123456789",
            "doctor": Doctor.objects.first().id,

            # "image": ("wwww-removebg-preview.png", open("C:/Users/DELL/Pictures/wwww-removebg-preview.png", "rb"))
        },
    )
    print(rp.data)
    assert rp.status_code == 201, rp.data
    assert "password" not in str(rp.data)


def test_login(client, db):
    test_create_user(client, db)
    rp = client.post(
        '/doctors/login/',
        format="json",
        data={
            "email_or_username": "om23440@gmail.com",
            "password": "123456789",
        }
    )
    assert rp.status_code == 200, rp.data
    assert "password" not in str(rp.data)
    assert "status" in rp.data and "message" in rp.data
    assert "token" in rp.data['data']