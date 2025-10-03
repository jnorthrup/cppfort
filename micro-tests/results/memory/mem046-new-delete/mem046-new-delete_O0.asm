
/Users/jim/work/cppfort/micro-tests/results/memory/mem046-new-delete/mem046-new-delete_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z15test_new_deletev>:
100000448: d100c3ff    	sub	sp, sp, #0x30
10000044c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000450: 910083fd    	add	x29, sp, #0x20
100000454: d2800080    	mov	x0, #0x4                ; =4
100000458: 9400001a    	bl	0x1000004c0 <__Znwm+0x1000004c0>
10000045c: 52800548    	mov	w8, #0x2a               ; =42
100000460: b9000008    	str	w8, [x0]
100000464: f81f83a0    	stur	x0, [x29, #-0x8]
100000468: f85f83a8    	ldur	x8, [x29, #-0x8]
10000046c: b9400108    	ldr	w8, [x8]
100000470: b81f43a8    	stur	w8, [x29, #-0xc]
100000474: f85f83a8    	ldur	x8, [x29, #-0x8]
100000478: f90007e8    	str	x8, [sp, #0x8]
10000047c: b40000a8    	cbz	x8, 0x100000490 <__Z15test_new_deletev+0x48>
100000480: 14000001    	b	0x100000484 <__Z15test_new_deletev+0x3c>
100000484: f94007e0    	ldr	x0, [sp, #0x8]
100000488: 94000011    	bl	0x1000004cc <__Znwm+0x1000004cc>
10000048c: 14000001    	b	0x100000490 <__Z15test_new_deletev+0x48>
100000490: b85f43a0    	ldur	w0, [x29, #-0xc]
100000494: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000498: 9100c3ff    	add	sp, sp, #0x30
10000049c: d65f03c0    	ret

00000001000004a0 <_main>:
1000004a0: d10083ff    	sub	sp, sp, #0x20
1000004a4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004a8: 910043fd    	add	x29, sp, #0x10
1000004ac: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000004b0: 97ffffe6    	bl	0x100000448 <__Z15test_new_deletev>
1000004b4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004b8: 910083ff    	add	sp, sp, #0x20
1000004bc: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004c0 <__stubs>:
1000004c0: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000004c4: f9400210    	ldr	x16, [x16]
1000004c8: d61f0200    	br	x16
1000004cc: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000004d0: f9400610    	ldr	x16, [x16, #0x8]
1000004d4: d61f0200    	br	x16
