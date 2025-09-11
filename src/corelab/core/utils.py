def try_delete(clazz, object_name):
    try:
        obj = clazz.get(object_name)
        obj.delete()

        # Check if wait_for_delete method exists and call it
        if hasattr(obj, 'wait_for_delete') and callable(getattr(obj, 'wait_for_delete')):
            obj.wait_for_delete()
        print(object_name, "deleted")
    except Exception as e:
        print("delete of", clazz, "failed:", e)
