
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf037-while-pointer/cf037-while-pointer_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z18test_while_pointerv>:
100000498: d10103ff    	sub	sp, sp, #0x40
10000049c: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000004a0: 9100c3fd    	add	x29, sp, #0x30
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 90000009    	adrp	x9, 0x100000000 <___stack_chk_guard+0x100000000>
1000004b8: 9115f129    	add	x9, x9, #0x57c
1000004bc: 3dc00120    	ldr	q0, [x9]
1000004c0: 910043e8    	add	x8, sp, #0x10
1000004c4: 3d8007e0    	str	q0, [sp, #0x10]
1000004c8: b9401129    	ldr	w9, [x9, #0x10]
1000004cc: b90023e9    	str	w9, [sp, #0x20]
1000004d0: f90007e8    	str	x8, [sp, #0x8]
1000004d4: b90007ff    	str	wzr, [sp, #0x4]
1000004d8: 14000001    	b	0x1000004dc <__Z18test_while_pointerv+0x44>
1000004dc: f94007e8    	ldr	x8, [sp, #0x8]
1000004e0: 910043e9    	add	x9, sp, #0x10
1000004e4: 91005129    	add	x9, x9, #0x14
1000004e8: eb090108    	subs	x8, x8, x9
1000004ec: 54000162    	b.hs	0x100000518 <__Z18test_while_pointerv+0x80>
1000004f0: 14000001    	b	0x1000004f4 <__Z18test_while_pointerv+0x5c>
1000004f4: f94007e8    	ldr	x8, [sp, #0x8]
1000004f8: b9400109    	ldr	w9, [x8]
1000004fc: b94007e8    	ldr	w8, [sp, #0x4]
100000500: 0b090108    	add	w8, w8, w9
100000504: b90007e8    	str	w8, [sp, #0x4]
100000508: f94007e8    	ldr	x8, [sp, #0x8]
10000050c: 91001108    	add	x8, x8, #0x4
100000510: f90007e8    	str	x8, [sp, #0x8]
100000514: 17fffff2    	b	0x1000004dc <__Z18test_while_pointerv+0x44>
100000518: b94007e8    	ldr	w8, [sp, #0x4]
10000051c: b90003e8    	str	w8, [sp]
100000520: f85f83a9    	ldur	x9, [x29, #-0x8]
100000524: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000528: f9400108    	ldr	x8, [x8]
10000052c: f9400108    	ldr	x8, [x8]
100000530: eb090108    	subs	x8, x8, x9
100000534: 54000060    	b.eq	0x100000540 <__Z18test_while_pointerv+0xa8>
100000538: 14000001    	b	0x10000053c <__Z18test_while_pointerv+0xa4>
10000053c: 9400000d    	bl	0x100000570 <___stack_chk_guard+0x100000570>
100000540: b94003e0    	ldr	w0, [sp]
100000544: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000548: 910103ff    	add	sp, sp, #0x40
10000054c: d65f03c0    	ret

0000000100000550 <_main>:
100000550: d10083ff    	sub	sp, sp, #0x20
100000554: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000558: 910043fd    	add	x29, sp, #0x10
10000055c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000560: 97ffffce    	bl	0x100000498 <__Z18test_while_pointerv>
100000564: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000568: 910083ff    	add	sp, sp, #0x20
10000056c: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000570 <__stubs>:
100000570: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100000574: f9400610    	ldr	x16, [x16, #0x8]
100000578: d61f0200    	br	x16
