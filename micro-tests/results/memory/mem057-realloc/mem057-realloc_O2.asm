
/Users/jim/work/cppfort/micro-tests/results/memory/mem057-realloc/mem057-realloc_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z12test_reallocv>:
100000448: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
10000044c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000450: 910043fd    	add	x29, sp, #0x10
100000454: 52800280    	mov	w0, #0x14               ; =20
100000458: 9400001d    	bl	0x1000004cc <_realloc+0x1000004cc>
10000045c: 52800548    	mov	w8, #0x2a               ; =42
100000460: b9000808    	str	w8, [x0, #0x8]
100000464: 52800501    	mov	w1, #0x28               ; =40
100000468: 9400001c    	bl	0x1000004d8 <_realloc+0x1000004d8>
10000046c: b9400813    	ldr	w19, [x0, #0x8]
100000470: 94000014    	bl	0x1000004c0 <_realloc+0x1000004c0>
100000474: aa1303e0    	mov	x0, x19
100000478: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000047c: a8c24ff4    	ldp	x20, x19, [sp], #0x20
100000480: d65f03c0    	ret

0000000100000484 <_main>:
100000484: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
100000488: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000048c: 910043fd    	add	x29, sp, #0x10
100000490: 52800280    	mov	w0, #0x14               ; =20
100000494: 9400000e    	bl	0x1000004cc <_realloc+0x1000004cc>
100000498: 52800548    	mov	w8, #0x2a               ; =42
10000049c: b9000808    	str	w8, [x0, #0x8]
1000004a0: 52800501    	mov	w1, #0x28               ; =40
1000004a4: 9400000d    	bl	0x1000004d8 <_realloc+0x1000004d8>
1000004a8: b9400813    	ldr	w19, [x0, #0x8]
1000004ac: 94000005    	bl	0x1000004c0 <_realloc+0x1000004c0>
1000004b0: aa1303e0    	mov	x0, x19
1000004b4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004b8: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000004bc: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004c0 <__stubs>:
1000004c0: 90000030    	adrp	x16, 0x100004000 <_realloc+0x100004000>
1000004c4: f9400210    	ldr	x16, [x16]
1000004c8: d61f0200    	br	x16
1000004cc: 90000030    	adrp	x16, 0x100004000 <_realloc+0x100004000>
1000004d0: f9400610    	ldr	x16, [x16, #0x8]
1000004d4: d61f0200    	br	x16
1000004d8: 90000030    	adrp	x16, 0x100004000 <_realloc+0x100004000>
1000004dc: f9400a10    	ldr	x16, [x16, #0x10]
1000004e0: d61f0200    	br	x16
