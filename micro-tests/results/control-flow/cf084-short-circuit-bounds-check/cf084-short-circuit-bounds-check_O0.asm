
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf084-short-circuit-bounds-check/cf084-short-circuit-bounds-check_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z17test_bounds_checkPiii>:
100000498: d10083ff    	sub	sp, sp, #0x20
10000049c: f9000be0    	str	x0, [sp, #0x10]
1000004a0: b9000fe1    	str	w1, [sp, #0xc]
1000004a4: b9000be2    	str	w2, [sp, #0x8]
1000004a8: b9400be8    	ldr	w8, [sp, #0x8]
1000004ac: 37f80248    	tbnz	w8, #0x1f, 0x1000004f4 <__Z17test_bounds_checkPiii+0x5c>
1000004b0: 14000001    	b	0x1000004b4 <__Z17test_bounds_checkPiii+0x1c>
1000004b4: b9400be8    	ldr	w8, [sp, #0x8]
1000004b8: b9400fe9    	ldr	w9, [sp, #0xc]
1000004bc: 6b090108    	subs	w8, w8, w9
1000004c0: 540001aa    	b.ge	0x1000004f4 <__Z17test_bounds_checkPiii+0x5c>
1000004c4: 14000001    	b	0x1000004c8 <__Z17test_bounds_checkPiii+0x30>
1000004c8: f9400be8    	ldr	x8, [sp, #0x10]
1000004cc: b9800be9    	ldrsw	x9, [sp, #0x8]
1000004d0: b8697908    	ldr	w8, [x8, x9, lsl #2]
1000004d4: 71000108    	subs	w8, w8, #0x0
1000004d8: 540000ed    	b.le	0x1000004f4 <__Z17test_bounds_checkPiii+0x5c>
1000004dc: 14000001    	b	0x1000004e0 <__Z17test_bounds_checkPiii+0x48>
1000004e0: f9400be8    	ldr	x8, [sp, #0x10]
1000004e4: b9800be9    	ldrsw	x9, [sp, #0x8]
1000004e8: b8697908    	ldr	w8, [x8, x9, lsl #2]
1000004ec: b9001fe8    	str	w8, [sp, #0x1c]
1000004f0: 14000004    	b	0x100000500 <__Z17test_bounds_checkPiii+0x68>
1000004f4: 12800008    	mov	w8, #-0x1               ; =-1
1000004f8: b9001fe8    	str	w8, [sp, #0x1c]
1000004fc: 14000001    	b	0x100000500 <__Z17test_bounds_checkPiii+0x68>
100000500: b9401fe0    	ldr	w0, [sp, #0x1c]
100000504: 910083ff    	add	sp, sp, #0x20
100000508: d65f03c0    	ret

000000010000050c <_main>:
10000050c: d10103ff    	sub	sp, sp, #0x40
100000510: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000514: 9100c3fd    	add	x29, sp, #0x30
100000518: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
10000051c: f9400108    	ldr	x8, [x8]
100000520: f9400108    	ldr	x8, [x8]
100000524: f81f83a8    	stur	x8, [x29, #-0x8]
100000528: b9000fff    	str	wzr, [sp, #0xc]
10000052c: 90000008    	adrp	x8, 0x100000000 <___stack_chk_guard+0x100000000>
100000530: 91165108    	add	x8, x8, #0x594
100000534: 3dc00100    	ldr	q0, [x8]
100000538: 910043e0    	add	x0, sp, #0x10
10000053c: 3d8007e0    	str	q0, [sp, #0x10]
100000540: b9401108    	ldr	w8, [x8, #0x10]
100000544: b90023e8    	str	w8, [sp, #0x20]
100000548: 528000a1    	mov	w1, #0x5                ; =5
10000054c: 52800042    	mov	w2, #0x2                ; =2
100000550: 97ffffd2    	bl	0x100000498 <__Z17test_bounds_checkPiii>
100000554: b9000be0    	str	w0, [sp, #0x8]
100000558: f85f83a9    	ldur	x9, [x29, #-0x8]
10000055c: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000560: f9400108    	ldr	x8, [x8]
100000564: f9400108    	ldr	x8, [x8]
100000568: eb090108    	subs	x8, x8, x9
10000056c: 54000060    	b.eq	0x100000578 <_main+0x6c>
100000570: 14000001    	b	0x100000574 <_main+0x68>
100000574: 94000005    	bl	0x100000588 <___stack_chk_guard+0x100000588>
100000578: b9400be0    	ldr	w0, [sp, #0x8]
10000057c: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000580: 910103ff    	add	sp, sp, #0x40
100000584: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000588 <__stubs>:
100000588: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000058c: f9400610    	ldr	x16, [x16, #0x8]
100000590: d61f0200    	br	x16
