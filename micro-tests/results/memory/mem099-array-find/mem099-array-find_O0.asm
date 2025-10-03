
/Users/jim/work/cppfort/micro-tests/results/memory/mem099-array-find/mem099-array-find_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z15test_array_findi>:
100000498: d10103ff    	sub	sp, sp, #0x40
10000049c: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000004a0: 9100c3fd    	add	x29, sp, #0x30
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: b9000be0    	str	w0, [sp, #0x8]
1000004b8: 90000008    	adrp	x8, 0x100000000 <___stack_chk_guard+0x100000000>
1000004bc: 91166108    	add	x8, x8, #0x598
1000004c0: 3dc00100    	ldr	q0, [x8]
1000004c4: 3d8007e0    	str	q0, [sp, #0x10]
1000004c8: b9401108    	ldr	w8, [x8, #0x10]
1000004cc: b90023e8    	str	w8, [sp, #0x20]
1000004d0: b90007ff    	str	wzr, [sp, #0x4]
1000004d4: 14000001    	b	0x1000004d8 <__Z15test_array_findi+0x40>
1000004d8: b94007e8    	ldr	w8, [sp, #0x4]
1000004dc: 71001508    	subs	w8, w8, #0x5
1000004e0: 5400022a    	b.ge	0x100000524 <__Z15test_array_findi+0x8c>
1000004e4: 14000001    	b	0x1000004e8 <__Z15test_array_findi+0x50>
1000004e8: b98007e9    	ldrsw	x9, [sp, #0x4]
1000004ec: 910043e8    	add	x8, sp, #0x10
1000004f0: b8697908    	ldr	w8, [x8, x9, lsl #2]
1000004f4: b9400be9    	ldr	w9, [sp, #0x8]
1000004f8: 6b090108    	subs	w8, w8, w9
1000004fc: 540000a1    	b.ne	0x100000510 <__Z15test_array_findi+0x78>
100000500: 14000001    	b	0x100000504 <__Z15test_array_findi+0x6c>
100000504: b94007e8    	ldr	w8, [sp, #0x4]
100000508: b9000fe8    	str	w8, [sp, #0xc]
10000050c: 14000009    	b	0x100000530 <__Z15test_array_findi+0x98>
100000510: 14000001    	b	0x100000514 <__Z15test_array_findi+0x7c>
100000514: b94007e8    	ldr	w8, [sp, #0x4]
100000518: 11000508    	add	w8, w8, #0x1
10000051c: b90007e8    	str	w8, [sp, #0x4]
100000520: 17ffffee    	b	0x1000004d8 <__Z15test_array_findi+0x40>
100000524: 12800008    	mov	w8, #-0x1               ; =-1
100000528: b9000fe8    	str	w8, [sp, #0xc]
10000052c: 14000001    	b	0x100000530 <__Z15test_array_findi+0x98>
100000530: b9400fe8    	ldr	w8, [sp, #0xc]
100000534: b90003e8    	str	w8, [sp]
100000538: f85f83a9    	ldur	x9, [x29, #-0x8]
10000053c: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000540: f9400108    	ldr	x8, [x8]
100000544: f9400108    	ldr	x8, [x8]
100000548: eb090108    	subs	x8, x8, x9
10000054c: 54000060    	b.eq	0x100000558 <__Z15test_array_findi+0xc0>
100000550: 14000001    	b	0x100000554 <__Z15test_array_findi+0xbc>
100000554: 9400000e    	bl	0x10000058c <___stack_chk_guard+0x10000058c>
100000558: b94003e0    	ldr	w0, [sp]
10000055c: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000560: 910103ff    	add	sp, sp, #0x40
100000564: d65f03c0    	ret

0000000100000568 <_main>:
100000568: d10083ff    	sub	sp, sp, #0x20
10000056c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000570: 910043fd    	add	x29, sp, #0x10
100000574: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000578: 52800060    	mov	w0, #0x3                ; =3
10000057c: 97ffffc7    	bl	0x100000498 <__Z15test_array_findi>
100000580: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000584: 910083ff    	add	sp, sp, #0x20
100000588: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

000000010000058c <__stubs>:
10000058c: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100000590: f9400610    	ldr	x16, [x16, #0x8]
100000594: d61f0200    	br	x16
