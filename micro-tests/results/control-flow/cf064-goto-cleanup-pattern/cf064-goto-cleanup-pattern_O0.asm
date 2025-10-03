
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf064-goto-cleanup-pattern/cf064-goto-cleanup-pattern_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z17test_goto_cleanupi>:
100000448: d100c3ff    	sub	sp, sp, #0x30
10000044c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000450: 910083fd    	add	x29, sp, #0x20
100000454: b81fc3a0    	stur	w0, [x29, #-0x4]
100000458: f9000bff    	str	xzr, [sp, #0x10]
10000045c: 12800008    	mov	w8, #-0x1               ; =-1
100000460: b9000fe8    	str	w8, [sp, #0xc]
100000464: b85fc3a8    	ldur	w8, [x29, #-0x4]
100000468: 36f80068    	tbz	w8, #0x1f, 0x100000474 <__Z17test_goto_cleanupi+0x2c>
10000046c: 14000001    	b	0x100000470 <__Z17test_goto_cleanupi+0x28>
100000470: 1400000e    	b	0x1000004a8 <__Z17test_goto_cleanupi+0x60>
100000474: d2800080    	mov	x0, #0x4                ; =4
100000478: 94000020    	bl	0x1000004f8 <__Znwm+0x1000004f8>
10000047c: 52800548    	mov	w8, #0x2a               ; =42
100000480: b9000008    	str	w8, [x0]
100000484: f9000be0    	str	x0, [sp, #0x10]
100000488: b85fc3a8    	ldur	w8, [x29, #-0x4]
10000048c: 35000068    	cbnz	w8, 0x100000498 <__Z17test_goto_cleanupi+0x50>
100000490: 14000001    	b	0x100000494 <__Z17test_goto_cleanupi+0x4c>
100000494: 14000005    	b	0x1000004a8 <__Z17test_goto_cleanupi+0x60>
100000498: f9400be8    	ldr	x8, [sp, #0x10]
10000049c: b9400108    	ldr	w8, [x8]
1000004a0: b9000fe8    	str	w8, [sp, #0xc]
1000004a4: 14000001    	b	0x1000004a8 <__Z17test_goto_cleanupi+0x60>
1000004a8: f9400be8    	ldr	x8, [sp, #0x10]
1000004ac: f90003e8    	str	x8, [sp]
1000004b0: b40000a8    	cbz	x8, 0x1000004c4 <__Z17test_goto_cleanupi+0x7c>
1000004b4: 14000001    	b	0x1000004b8 <__Z17test_goto_cleanupi+0x70>
1000004b8: f94003e0    	ldr	x0, [sp]
1000004bc: 94000012    	bl	0x100000504 <__Znwm+0x100000504>
1000004c0: 14000001    	b	0x1000004c4 <__Z17test_goto_cleanupi+0x7c>
1000004c4: b9400fe0    	ldr	w0, [sp, #0xc]
1000004c8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000004cc: 9100c3ff    	add	sp, sp, #0x30
1000004d0: d65f03c0    	ret

00000001000004d4 <_main>:
1000004d4: d10083ff    	sub	sp, sp, #0x20
1000004d8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004dc: 910043fd    	add	x29, sp, #0x10
1000004e0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000004e4: 52800020    	mov	w0, #0x1                ; =1
1000004e8: 97ffffd8    	bl	0x100000448 <__Z17test_goto_cleanupi>
1000004ec: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004f0: 910083ff    	add	sp, sp, #0x20
1000004f4: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004f8 <__stubs>:
1000004f8: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000004fc: f9400210    	ldr	x16, [x16]
100000500: d61f0200    	br	x16
100000504: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000508: f9400610    	ldr	x16, [x16, #0x8]
10000050c: d61f0200    	br	x16
