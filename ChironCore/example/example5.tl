 :x = 20
 :y = 10
 :a = 10
 forward :x
 if (:y > :x) [
  :x = 30
  :y = 0
   left :y
 ] else [
  :x = 25
   right 50
 ]
 forward :x+:a
 forward :y