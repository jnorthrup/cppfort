
/Users/jim/work/cppfort/micro-tests/results/memory/mem054-delete-null/mem054-delete-null_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z16test_delete_nullv>:
100000448: d10083ff    	sub	sp, sp, #0x20
10000044c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000450: 910043fd    	add	x29, sp, #0x10
100000454: f90007ff    	str	xzr, [sp, #0x8]
100000458: f94007e8    	ldr	x8, [sp, #0x8]
10000045c: f90003e8    	str	x8, [sp]
100000460: b40000a8    	cbz	x8, 0x100000474 <__Z16test_delete_nullv+0x2c>
100000464: 14000001    	b	0x100000468 <__Z16test_delete_nullv+0x20>
100000468: f94003e0    	ldr	x0, [sp]
10000046c: 9400000e    	bl	0x1000004a4 <__ZdlPv+0x1000004a4>
100000470: 14000001    	b	0x100000474 <__Z16test_delete_nullv+0x2c>
100000474: 52800000    	mov	w0, #0x0                ; =0
100000478: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000047c: 910083ff    	add	sp, sp, #0x20
100000480: d65f03c0    	ret

0000000100000484 <_main>:
100000484: d10083ff    	sub	sp, sp, #0x20
100000488: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000048c: 910043fd    	add	x29, sp, #0x10
100000490: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000494: 97ffffed    	bl	0x100000448 <__Z16test_delete_nullv>
100000498: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000049c: 910083ff    	add	sp, sp, #0x20
1000004a0: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004a4 <__stubs>:
1000004a4: 90000030    	adrp	x16, 0x100004000 <__ZdlPv+0x100004000>
1000004a8: f9400210    	ldr	x16, [x16]
1000004ac: d61f0200    	br	x16
