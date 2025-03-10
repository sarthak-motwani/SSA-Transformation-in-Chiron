:x = 50
:z = :x + 7
if (:z < 100) [
  :z = 5
]
backward :x + :z
forward :z