
/Users/jim/work/cppfort/micro-tests/results/memory/mem070-dynamic-2d-array/mem070-dynamic-2d-array_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z21test_dynamic_2d_arrayv>:
100000448: d10103ff    	sub	sp, sp, #0x40
10000044c: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000450: 9100c3fd    	add	x29, sp, #0x30
100000454: d2800300    	mov	x0, #0x18               ; =24
100000458: 94000041    	bl	0x10000055c <__Znam+0x10000055c>
10000045c: f81f83a0    	stur	x0, [x29, #-0x8]
100000460: b81f43bf    	stur	wzr, [x29, #-0xc]
100000464: 14000001    	b	0x100000468 <__Z21test_dynamic_2d_arrayv+0x20>
100000468: b85f43a8    	ldur	w8, [x29, #-0xc]
10000046c: 71000d08    	subs	w8, w8, #0x3
100000470: 5400018a    	b.ge	0x1000004a0 <__Z21test_dynamic_2d_arrayv+0x58>
100000474: 14000001    	b	0x100000478 <__Z21test_dynamic_2d_arrayv+0x30>
100000478: d2800180    	mov	x0, #0xc                ; =12
10000047c: 94000038    	bl	0x10000055c <__Znam+0x10000055c>
100000480: f85f83a8    	ldur	x8, [x29, #-0x8]
100000484: b89f43a9    	ldursw	x9, [x29, #-0xc]
100000488: f8297900    	str	x0, [x8, x9, lsl #3]
10000048c: 14000001    	b	0x100000490 <__Z21test_dynamic_2d_arrayv+0x48>
100000490: b85f43a8    	ldur	w8, [x29, #-0xc]
100000494: 11000508    	add	w8, w8, #0x1
100000498: b81f43a8    	stur	w8, [x29, #-0xc]
10000049c: 17fffff3    	b	0x100000468 <__Z21test_dynamic_2d_arrayv+0x20>
1000004a0: f85f83a8    	ldur	x8, [x29, #-0x8]
1000004a4: f9400509    	ldr	x9, [x8, #0x8]
1000004a8: 52800548    	mov	w8, #0x2a               ; =42
1000004ac: b9000528    	str	w8, [x9, #0x4]
1000004b0: f85f83a8    	ldur	x8, [x29, #-0x8]
1000004b4: f9400508    	ldr	x8, [x8, #0x8]
1000004b8: b9400508    	ldr	w8, [x8, #0x4]
1000004bc: b81f03a8    	stur	w8, [x29, #-0x10]
1000004c0: b81ec3bf    	stur	wzr, [x29, #-0x14]
1000004c4: 14000001    	b	0x1000004c8 <__Z21test_dynamic_2d_arrayv+0x80>
1000004c8: b85ec3a8    	ldur	w8, [x29, #-0x14]
1000004cc: 71000d08    	subs	w8, w8, #0x3
1000004d0: 5400020a    	b.ge	0x100000510 <__Z21test_dynamic_2d_arrayv+0xc8>
1000004d4: 14000001    	b	0x1000004d8 <__Z21test_dynamic_2d_arrayv+0x90>
1000004d8: f85f83a8    	ldur	x8, [x29, #-0x8]
1000004dc: b89ec3a9    	ldursw	x9, [x29, #-0x14]
1000004e0: f8697908    	ldr	x8, [x8, x9, lsl #3]
1000004e4: f9000be8    	str	x8, [sp, #0x10]
1000004e8: b40000a8    	cbz	x8, 0x1000004fc <__Z21test_dynamic_2d_arrayv+0xb4>
1000004ec: 14000001    	b	0x1000004f0 <__Z21test_dynamic_2d_arrayv+0xa8>
1000004f0: f9400be0    	ldr	x0, [sp, #0x10]
1000004f4: 9400001d    	bl	0x100000568 <__Znam+0x100000568>
1000004f8: 14000001    	b	0x1000004fc <__Z21test_dynamic_2d_arrayv+0xb4>
1000004fc: 14000001    	b	0x100000500 <__Z21test_dynamic_2d_arrayv+0xb8>
100000500: b85ec3a8    	ldur	w8, [x29, #-0x14]
100000504: 11000508    	add	w8, w8, #0x1
100000508: b81ec3a8    	stur	w8, [x29, #-0x14]
10000050c: 17ffffef    	b	0x1000004c8 <__Z21test_dynamic_2d_arrayv+0x80>
100000510: f85f83a8    	ldur	x8, [x29, #-0x8]
100000514: f90007e8    	str	x8, [sp, #0x8]
100000518: b40000a8    	cbz	x8, 0x10000052c <__Z21test_dynamic_2d_arrayv+0xe4>
10000051c: 14000001    	b	0x100000520 <__Z21test_dynamic_2d_arrayv+0xd8>
100000520: f94007e0    	ldr	x0, [sp, #0x8]
100000524: 94000011    	bl	0x100000568 <__Znam+0x100000568>
100000528: 14000001    	b	0x10000052c <__Z21test_dynamic_2d_arrayv+0xe4>
10000052c: b85f03a0    	ldur	w0, [x29, #-0x10]
100000530: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000534: 910103ff    	add	sp, sp, #0x40
100000538: d65f03c0    	ret

000000010000053c <_main>:
10000053c: d10083ff    	sub	sp, sp, #0x20
100000540: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000544: 910043fd    	add	x29, sp, #0x10
100000548: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000054c: 97ffffbf    	bl	0x100000448 <__Z21test_dynamic_2d_arrayv>
100000550: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000554: 910083ff    	add	sp, sp, #0x20
100000558: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

000000010000055c <__stubs>:
10000055c: 90000030    	adrp	x16, 0x100004000 <__Znam+0x100004000>
100000560: f9400210    	ldr	x16, [x16]
100000564: d61f0200    	br	x16
100000568: 90000030    	adrp	x16, 0x100004000 <__Znam+0x100004000>
10000056c: f9400610    	ldr	x16, [x16, #0x8]
100000570: d61f0200    	br	x16
