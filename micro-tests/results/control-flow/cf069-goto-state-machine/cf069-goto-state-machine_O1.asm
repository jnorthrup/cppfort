
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf069-goto-state-machine/cf069-goto-state-machine_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z23test_goto_state_machinei>:
100000360: 52800288    	mov	w8, #0x14               ; =20
100000364: 7100001f    	cmp	w0, #0x0
100000368: 1a9f0100    	csel	w0, w8, wzr, eq
10000036c: d65f03c0    	ret

0000000100000370 <_main>:
100000370: 52800280    	mov	w0, #0x14               ; =20
100000374: d65f03c0    	ret
