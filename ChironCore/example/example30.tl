:base = 120
:angle = 90
:depth = 4
:step = 0
:dir = 1
:shape = 1
:accel = 2
:decay = 1
:counter = 0
:limit = 6
:alternate = 0
:geofactor = 3
:spiralbase = 10

pendown
repeat 5 [
    if (:shape * :dir > :limit) [
        :shape = 1
        :dir = :dir * -1
        :alternate = :alternate + 1
    ] else [
        :shape = :shape + :accel
        :accel = :accel + :decay
    ]

    if (:alternate < 2) [
        repeat :shape [
            forward :base
            right (360 / :shape)
            :step = :step + 1
            
            repeat 2 [
                if (:step > 3) [
                    penup
                    goto (:base * :geofactor, :base / :geofactor)
                    pendown
                    :geofactor = :geofactor + 1
                    
                    repeat 4 [
                        left 90
                        forward (:base - :counter)
                        :counter = :counter + 5
                        if (:counter > 40) [
                            :counter = 0
                            right 45
                            forward (:base / 2)
                        ]
                    ]
                ] else [
                    backward (:base / 2)
                    left 60
                    :dir = :dir + 1
                ]
            ]
        ]
    ] else [
        :spiralsteps = :spiralbase
        :spiralangle = 15
        
        repeat 3 [
            forward :spiralsteps
            right :spiralangle
            :spiralsteps = :spiralsteps * 2
            :spiralangle = :spiralangle + 5
            
            if (:spiralsteps > 50) [
                penup
                goto (:spiralsteps, :spiralangle)
                pendown
                repeat 2 [
                    left 135
                    forward (:spiralsteps / 2)
                ]
            ]
        ]
    ]

    :temp = :base
    :base = :base + (:counter * 2)
    :counter = :temp - (:step / 2)
    :limit = :limit + (:shape * :dir)

    if (:base > 200) [
        :base = 60
        :dir = :dir * -1
        repeat 2 [
            right 180
            forward 30
            :geofactor = :geofactor - 1
        ]
    ] else [
        left 45
        forward (:base * 3 / 4)
    ]
    
    penup
    goto (0, 0)
    pendown
    :depth = :depth - 1
]
penup