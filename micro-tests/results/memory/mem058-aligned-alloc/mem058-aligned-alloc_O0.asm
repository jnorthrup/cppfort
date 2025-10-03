
/Users/jim/work/cppfort/micro-tests/results/memory/mem058-aligned-alloc/mem058-aligned-alloc_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z18test_aligned_allocv>:
100000448: d100c3ff    	sub	sp, sp, #0x30
10000044c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000450: 910083fd    	add	x29, sp, #0x20
100000454: d2800800    	mov	x0, #0x40               ; =64
100000458: d2800081    	mov	x1, #0x4                ; =4
10000045c: 9400001f    	bl	0x1000004d8 <_free+0x1000004d8>
100000460: f9000be0    	str	x0, [sp, #0x10]
100000464: f9400be8    	ldr	x8, [sp, #0x10]
100000468: b50000a8    	cbnz	x8, 0x10000047c <__Z18test_aligned_allocv+0x34>
10000046c: 14000001    	b	0x100000470 <__Z18test_aligned_allocv+0x28>
100000470: 12800008    	mov	w8, #-0x1               ; =-1
100000474: b81fc3a8    	stur	w8, [x29, #-0x4]
100000478: 1400000c    	b	0x1000004a8 <__Z18test_aligned_allocv+0x60>
10000047c: f9400be9    	ldr	x9, [sp, #0x10]
100000480: 52800548    	mov	w8, #0x2a               ; =42
100000484: b9000128    	str	w8, [x9]
100000488: f9400be8    	ldr	x8, [sp, #0x10]
10000048c: b9400108    	ldr	w8, [x8]
100000490: b9000fe8    	str	w8, [sp, #0xc]
100000494: f9400be0    	ldr	x0, [sp, #0x10]
100000498: 94000013    	bl	0x1000004e4 <_free+0x1000004e4>
10000049c: b9400fe8    	ldr	w8, [sp, #0xc]
1000004a0: b81fc3a8    	stur	w8, [x29, #-0x4]
1000004a4: 14000001    	b	0x1000004a8 <__Z18test_aligned_allocv+0x60>
1000004a8: b85fc3a0    	ldur	w0, [x29, #-0x4]
1000004ac: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000004b0: 9100c3ff    	add	sp, sp, #0x30
1000004b4: d65f03c0    	ret

00000001000004b8 <_main>:
1000004b8: d10083ff    	sub	sp, sp, #0x20
1000004bc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004c0: 910043fd    	add	x29, sp, #0x10
1000004c4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000004c8: 97ffffe0    	bl	0x100000448 <__Z18test_aligned_allocv>
1000004cc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004d0: 910083ff    	add	sp, sp, #0x20
1000004d4: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004d8 <__stubs>:
1000004d8: 90000030    	adrp	x16, 0x100004000 <_free+0x100004000>
1000004dc: f9400210    	ldr	x16, [x16]
1000004e0: d61f0200    	br	x16
1000004e4: 90000030    	adrp	x16, 0x100004000 <_free+0x100004000>
1000004e8: f9400610    	ldr	x16, [x16, #0x8]
1000004ec: d61f0200    	br	x16
