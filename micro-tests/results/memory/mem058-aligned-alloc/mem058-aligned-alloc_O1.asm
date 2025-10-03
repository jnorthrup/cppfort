
/Users/jim/work/cppfort/micro-tests/results/memory/mem058-aligned-alloc/mem058-aligned-alloc_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z18test_aligned_allocv>:
100000448: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
10000044c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000450: 910043fd    	add	x29, sp, #0x10
100000454: 52800800    	mov	w0, #0x40               ; =64
100000458: 52800081    	mov	w1, #0x4                ; =4
10000045c: 9400001b    	bl	0x1000004c8 <_free+0x1000004c8>
100000460: b40000a0    	cbz	x0, 0x100000474 <__Z18test_aligned_allocv+0x2c>
100000464: 52800553    	mov	w19, #0x2a              ; =42
100000468: b9000013    	str	w19, [x0]
10000046c: 9400001a    	bl	0x1000004d4 <_free+0x1000004d4>
100000470: 14000002    	b	0x100000478 <__Z18test_aligned_allocv+0x30>
100000474: 12800013    	mov	w19, #-0x1              ; =-1
100000478: aa1303e0    	mov	x0, x19
10000047c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000480: a8c24ff4    	ldp	x20, x19, [sp], #0x20
100000484: d65f03c0    	ret

0000000100000488 <_main>:
100000488: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
10000048c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000490: 910043fd    	add	x29, sp, #0x10
100000494: 52800800    	mov	w0, #0x40               ; =64
100000498: 52800081    	mov	w1, #0x4                ; =4
10000049c: 9400000b    	bl	0x1000004c8 <_free+0x1000004c8>
1000004a0: b40000a0    	cbz	x0, 0x1000004b4 <_main+0x2c>
1000004a4: 52800553    	mov	w19, #0x2a              ; =42
1000004a8: b9000013    	str	w19, [x0]
1000004ac: 9400000a    	bl	0x1000004d4 <_free+0x1000004d4>
1000004b0: 14000002    	b	0x1000004b8 <_main+0x30>
1000004b4: 12800013    	mov	w19, #-0x1              ; =-1
1000004b8: aa1303e0    	mov	x0, x19
1000004bc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004c0: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000004c4: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004c8 <__stubs>:
1000004c8: 90000030    	adrp	x16, 0x100004000 <_free+0x100004000>
1000004cc: f9400210    	ldr	x16, [x16]
1000004d0: d61f0200    	br	x16
1000004d4: 90000030    	adrp	x16, 0x100004000 <_free+0x100004000>
1000004d8: f9400610    	ldr	x16, [x16, #0x8]
1000004dc: d61f0200    	br	x16
