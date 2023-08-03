from pynput.keyboard import Key, Listener, KeyCode

def on_press(key):
    print('{0} pressed'.format(key))
    print(key)
    if key in [Key.up, Key.down, Key.left, Key.right, KeyCode.from_char('a'), KeyCode.from_char('w'),
               KeyCode.from_char('s'), KeyCode.from_char('d')]:
        print('Yes! Move')

def on_release(key):
    print('{0} release'.format(
        key))
    if key == Key.esc:
        # Stop listener
        return False

# Collect events until released
with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

# listener = Listener(on_press=on_press, on_release=on_release)
# listener.start()