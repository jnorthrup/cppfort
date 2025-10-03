
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf048-do-while-assignment/cf048-do-while-assignment_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z9get_valueRi>:
100000360: b9400008    	ldr	w8, [x0]
100000364: 11000509    	add	w9, w8, #0x1
100000368: b9000009    	str	w9, [x0]
10000036c: aa0803e0    	mov	x0, x8
100000370: d65f03c0    	ret

0000000100000374 <__Z24test_do_while_assignmentv>:
100000374: 528006e0    	mov	w0, #0x37               ; =55
100000378: d65f03c0    	ret

000000010000037c <_main>:
10000037c: 528006e0    	mov	w0, #0x37               ; =55
100000380: d65f03c0    	ret
