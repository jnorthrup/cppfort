
/Users/jim/work/cppfort/micro-tests/results/memory/mem070-dynamic-2d-array/mem070-dynamic-2d-array_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z21test_dynamic_2d_arrayv>:
100000448: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
10000044c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000450: 910043fd    	add	x29, sp, #0x10
100000454: 52800300    	mov	w0, #0x18               ; =24
100000458: 9400003d    	bl	0x10000054c <__Znam+0x10000054c>
10000045c: aa0003f3    	mov	x19, x0
100000460: d2800014    	mov	x20, #0x0               ; =0
100000464: 52800180    	mov	w0, #0xc                ; =12
100000468: 94000039    	bl	0x10000054c <__Znam+0x10000054c>
10000046c: f8346a60    	str	x0, [x19, x20]
100000470: 91002294    	add	x20, x20, #0x8
100000474: f100629f    	cmp	x20, #0x18
100000478: 54ffff61    	b.ne	0x100000464 <__Z21test_dynamic_2d_arrayv+0x1c>
10000047c: d2800014    	mov	x20, #0x0               ; =0
100000480: f9400668    	ldr	x8, [x19, #0x8]
100000484: 52800549    	mov	w9, #0x2a               ; =42
100000488: b9000509    	str	w9, [x8, #0x4]
10000048c: 14000004    	b	0x10000049c <__Z21test_dynamic_2d_arrayv+0x54>
100000490: 91002294    	add	x20, x20, #0x8
100000494: f100629f    	cmp	x20, #0x18
100000498: 540000a0    	b.eq	0x1000004ac <__Z21test_dynamic_2d_arrayv+0x64>
10000049c: f8746a60    	ldr	x0, [x19, x20]
1000004a0: b4ffff80    	cbz	x0, 0x100000490 <__Z21test_dynamic_2d_arrayv+0x48>
1000004a4: 94000027    	bl	0x100000540 <__Znam+0x100000540>
1000004a8: 17fffffa    	b	0x100000490 <__Z21test_dynamic_2d_arrayv+0x48>
1000004ac: aa1303e0    	mov	x0, x19
1000004b0: 94000024    	bl	0x100000540 <__Znam+0x100000540>
1000004b4: 52800540    	mov	w0, #0x2a               ; =42
1000004b8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004bc: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000004c0: d65f03c0    	ret

00000001000004c4 <_main>:
1000004c4: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
1000004c8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004cc: 910043fd    	add	x29, sp, #0x10
1000004d0: 52800300    	mov	w0, #0x18               ; =24
1000004d4: 9400001e    	bl	0x10000054c <__Znam+0x10000054c>
1000004d8: aa0003f3    	mov	x19, x0
1000004dc: d2800014    	mov	x20, #0x0               ; =0
1000004e0: 52800180    	mov	w0, #0xc                ; =12
1000004e4: 9400001a    	bl	0x10000054c <__Znam+0x10000054c>
1000004e8: f8346a60    	str	x0, [x19, x20]
1000004ec: 91002294    	add	x20, x20, #0x8
1000004f0: f100629f    	cmp	x20, #0x18
1000004f4: 54ffff61    	b.ne	0x1000004e0 <_main+0x1c>
1000004f8: d2800014    	mov	x20, #0x0               ; =0
1000004fc: f9400668    	ldr	x8, [x19, #0x8]
100000500: 52800549    	mov	w9, #0x2a               ; =42
100000504: b9000509    	str	w9, [x8, #0x4]
100000508: 14000004    	b	0x100000518 <_main+0x54>
10000050c: 91002294    	add	x20, x20, #0x8
100000510: f100629f    	cmp	x20, #0x18
100000514: 540000a0    	b.eq	0x100000528 <_main+0x64>
100000518: f8746a60    	ldr	x0, [x19, x20]
10000051c: b4ffff80    	cbz	x0, 0x10000050c <_main+0x48>
100000520: 94000008    	bl	0x100000540 <__Znam+0x100000540>
100000524: 17fffffa    	b	0x10000050c <_main+0x48>
100000528: aa1303e0    	mov	x0, x19
10000052c: 94000005    	bl	0x100000540 <__Znam+0x100000540>
100000530: 52800540    	mov	w0, #0x2a               ; =42
100000534: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000538: a8c24ff4    	ldp	x20, x19, [sp], #0x20
10000053c: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000540 <__stubs>:
100000540: 90000030    	adrp	x16, 0x100004000 <__Znam+0x100004000>
100000544: f9400210    	ldr	x16, [x16]
100000548: d61f0200    	br	x16
10000054c: 90000030    	adrp	x16, 0x100004000 <__Znam+0x100004000>
100000550: f9400610    	ldr	x16, [x16, #0x8]
100000554: d61f0200    	br	x16
