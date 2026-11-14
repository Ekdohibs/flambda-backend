(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* The check of the regions of one term: a test of the term alone, with
   a certificate, that its regions add no fault to it on the machine
   of regions (FlambdaRegionsSpec.v).

   This file is to be reviewed: all of it. It adds nothing to the
   machine: [region_step] and the rules of FlambdaRegionsSpec.v keep
   their text. Nothing proves the statement at the end yet: the file
   is here for the review of what it says, with
   FlambdaRegionsCheckedSpec.v, which has the property of states that
   the proof is to go by.

   WHAT IT IS FOR

   The validation of a run of Simplify compares the term before and
   the term after with their regions dropped
   (FlambdaRegionsSpec.erase). It says nothing of the regions of the
   term after. This check is of the term after alone. If it passes,
   the regions of the term add no fault to it
   ([regions_add_no_fault]): where the run of the term on the machine
   of regions stops without a step, the run of the term with its
   regions dropped goes wrong too. That is an implication and no
   more: of a term that goes wrong with its regions dropped it says
   nothing. So a term that passes and that does not go wrong with its
   regions dropped uses no value whose region is gone (decision D45),
   makes no heap block that holds a local value (decision D46), ends
   or names no region that is not open (decision D47), and has no
   primitive of regions that is not as the conversion makes it. With
   the validation of today, a term before that does not go wrong,
   regions dropped, gives a term after that never stops without a
   step, regions and all ([checked_after_safe_statement]). That no
   state of a run of a term that passes stops for one of those
   reasons, whatever the term does with its regions dropped, is a
   statement of FlambdaRegionsCheckedSpec.v
   ([checked_fault_statement], with the two before it).

   The check is a function. It takes a certificate, which an analysis
   that is not trusted guesses: nothing is asked of the certificate
   but that the check passes with it.

   WHAT THE CHECK KNOWS

   Of a variable, how long its value lives ([lifetime]): for ever; as
   long as the code that was given it runs; as long as a region is
   open; not at all. Of a point of the term, the regions that are
   open there and that the term can name, newest first ([cs_open]).
   The machine may have other frames between them: regions that
   nothing names any more, and the frames of the callers. A value in
   such a frame lives as long as the nearest region below it that has
   a name, because only the end of a region at or below a frame pops
   the frame.

   A local allocation goes in the newest frame, whatever region it
   names (decision D44), so it is given the lifetime of the newest
   region that has a name, or that of the code if none has
   ([newest_lifetime]). That is the longest that can be said of it.

   THE CERTIFICATE

   - For each continuation that a [Let_cont] binds: the regions that
     are open when its handler is entered, by the names that the
     handler has for them (a parameter of the kind [Region], or a
     region that is open where the handler is bound), and the lifetime
     of each parameter. A continuation with no entry is one that
     nothing may jump to.
   - For each code: which parameters may be given a local value,
     whether a closure of it may be local, whether it may return a
     local value.
   - Whether a callee that is not known may be given a local value,
     whether it may return one, and whether a partial application may
     keep one.

   The check verifies each handler and each code against its entry,
   and each jump and each call against the entry of what it goes to.

   A JUMP

   A handler is entered with regions of its own choice among those
   that are open at the jump, in their order ([in_order]): a region
   that is open at the jump and not among them is not ended by name
   after it, and what is in it lives as long as the nearest region
   below it that the handler does enter with ([nearest_named]), or as
   long as the code. An argument must live as long as the handler
   takes its parameter to ([outlives]).

   A CALL

   The machine has no stack of calls. Code is entered with no region
   open, its parameters and its closure living for ever or as long as
   the code (the certificate says which may be local). What it
   returns lives for ever, or, if the certificate says that the code
   may return a local value, as long as the newest region of the
   caller that has a name.

   A callee that is not known is any code, or a partial application.
   The machine runs an application with too many arguments as a full
   call whose result is called with the rest ([Over_application]),
   and a call with too few makes a partial application, a value that
   keeps the callee and the arguments and that the machine never
   takes to be local. [unknown_call] says what such a call may
   return, over every way in which the arguments may be split between
   the calls; if the certificate says that no partial application
   keeps a local value, it refuses a call that could make one that
   does.

   WHAT THE CHECK DOES NOT DO

   - It does not say that a region is ended. Code may leave a frame
     that it pushed, by a return or by a raise (decision D48); the
     check passes such a term.
   - It says nothing of the kinds of values but where the machine of
     regions has a rule of its own: a [Let] of a primitive of regions,
     a [Let] that is a local allocation, a region given to a parameter.
     A term that the check passes may stop without a step for a
     reason that it has with its regions dropped.
   - It says nothing of the primitives of the heap: the machine of
     regions runs none ([FlambdaRegionsSpec.plain_step] has no step
     for one), and a term that has one is passed or refused by the
     lifetimes of its arguments alone.
   - It is of the model's term on the model's machine. Two places
     where that machine keeps a value longer than the program does
     are known, and the check follows the machine in both: an
     application with too many arguments of a function that is not
     known, where the helper that the backend makes begins and ends a
     region of its own (FlambdaRegionsSpec.v, "what this stage leaves
     out"); and a partial application of a function that is not
     known, whose closure the program may allocate locally and the
     machine never does. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaRegionsSpec FlambdaRegionsErasureSpec.
Import ListNotations.

(* ======================================================================== *)
(* Lifetimes                                                                *)
(* ======================================================================== *)

(* How long the value of a variable lives: for ever (it is not local,
   and nothing that it reaches is); as long as the code that was given
   it runs; as long as a region is open; not at all (the variable may
   not be evaluated: the variable of a region, or a value that the
   check has lost track of). *)
Inductive lifetime : Type :=
  | Forever
  | Of_caller
  | Of_region (r : variable)
  | Gone.

Definition is_forever (l : lifetime) : bool :=
  match l with
  | Forever => true
  | Of_caller | Of_region _ | Gone => false
  end.

(* The open regions down to [r], [r] included; all of them if [r] is
   not open. *)
Fixpoint down_to (r : variable) (open : list variable) : list variable :=
  match open with
  | [] => []
  | x :: open => if Pos.eqb x r then [x] else x :: down_to r open
  end.

(* The open regions below [r]; none if [r] is not open. *)
Fixpoint regions_below (r : variable) (open : list variable)
    : list variable :=
  match open with
  | [] => []
  | x :: open => if Pos.eqb x r then open else regions_below r open
  end.

(* [xs] in the order of [ys], which may have others between. *)
Fixpoint in_order (xs ys : list variable) : bool :=
  match ys with
  | [] => match xs with [] => true | _ :: _ => false end
  | y :: ys' =>
    match xs with
    | [] => true
    | x :: xs' => if Pos.eqb x y then in_order xs' ys' else in_order xs ys'
    end
  end.

(* Of two lifetimes, the one that ends first. Of two regions it is the
   newer; of two regions one of which is not open, nothing is known. *)
Definition shorter (open : list variable) (a b : lifetime) : lifetime :=
  match a, b with
  | Gone, _ | _, Gone => Gone
  | Forever, l | l, Forever => l
  | Of_caller, l | l, Of_caller => l
  | Of_region r, Of_region r' =>
    if among r open && among r' open
    then if among r (down_to r' open) then a else b
    else Gone
  end.

(* ======================================================================== *)
(* The certificate                                                          *)
(* ======================================================================== *)

(* A handler: the regions that are open when it is entered, by the
   names that it has for them (a parameter that is a region, or a
   region that was open where the handler is bound), newest first;
   and the lifetime of each parameter, in order. *)
Record handler_entry : Type := Handler_entry {
  he_open : list variable;
  he_params : list lifetime
}.

(* Code: for each parameter whether it may be given a local value;
   whether a closure of it may be local; whether it may return a local
   value. *)
Record code_entry : Type := Code_entry {
  ce_params_local : list bool;
  ce_closure_local : bool;
  ce_returns_local : bool
}.

Record certificate : Type := Certificate {
  cert_handlers : list (continuation * handler_entry);
  cert_code : list (code_id * code_entry);
  (* A callee that is not known may be given a local value. *)
  cert_indirect_local : bool;
  (* A callee that is not known may return one. *)
  cert_unknown_returns_local : bool;
  (* A partial application may keep a local value. If not, none does. *)
  cert_partial_local : bool
}.

(* ======================================================================== *)
(* The state of the check                                                   *)
(* ======================================================================== *)

(* What a continuation in scope is: a handler with its entry; a
   handler that the certificate has no entry for, which nothing may
   jump to; the return continuation of the code, with whether the code
   may return a local value; its exception continuation. A
   continuation that is none of them is a way out of the term. *)
Inductive cont_entry : Type :=
  | To_handler (params : list variable) (kinds : list flambda_kind)
      (entry : handler_entry)
  | Unreachable
  | To_return (may_be_local : bool)
  | To_raise.

(* The lifetimes of the variables; the regions that are open and have
   a name, newest first; every variable of a region in scope, open or
   not, none of which may be bound again; the continuations. *)
Record check_state : Type := Check_state {
  cs_env : list (variable * lifetime);
  cs_open : list variable;
  cs_regions : list variable;
  cs_conts : list (continuation * cont_entry)
}.

Definition bind_lifetime (x : variable) (l : lifetime) (cs : check_state)
    : check_state :=
  Check_state ((x, l) :: cs_env cs) (cs_open cs) (cs_regions cs) (cs_conts cs).

(* A variable that may be bound: not the variable of a region. *)
Definition not_a_region (cs : check_state) (x : variable) : bool :=
  negb (among x (cs_regions cs)).

Definition lifetime_of (cs : check_state) (s : simple) : lifetime :=
  match s with
  | Var x =>
    match lookup x (cs_env cs) with
    | Some l => l
    | None => Gone
    end
  | Symbol _ | Const _ => Forever
  end.

(* Decision D45: a value that may be evaluated. *)
Definition lifetime_usable (cs : check_state) (l : lifetime) : bool :=
  match l with
  | Forever | Of_caller => true
  | Of_region r => among r (cs_open cs)
  | Gone => false
  end.

Definition check_usable (cs : check_state) (s : simple) : bool :=
  lifetime_usable cs (lifetime_of cs s).

(* The lifetime of a local allocation: that of the newest region that
   is open and has a name; with none, that of the code. *)
Definition newest_lifetime (cs : check_state) : lifetime :=
  match cs_open cs with
  | r :: _ => Of_region r
  | [] => Of_caller
  end.

Definition shortest (cs : check_state) (ls : list lifetime) : lifetime :=
  fold_right (shorter (cs_open cs)) Forever ls.

(* The lifetime of what a [Let] binds that does not allocate. What is
   read from a value lives as long as it; a field that the access says
   is an immediate is one, or the load has no value; so is an element
   that the load takes as an immediate. An element of an array lives
   as long as the array: the index is a number. Every other primitive
   that has a value on this machine gives a number. *)
Definition named_lifetime (cs : check_state) (defining_expr : named)
    : lifetime :=
  match defining_expr with
  | Simple s => lifetime_of cs s
  | Prim (Unary (Block_load (Block_access_values _ _ Immediate) _ _)) _ =>
    Forever
  | Prim (Binary (Array_load _ Load_immediates _)) _ => Forever
  | Prim (Binary (Array_load _ _ _)) [array; _] => lifetime_of cs array
  | Prim (Unary (Block_load _ _ _ | Opaque_identity _)) args
  | Prim (Binary (Array_load _ _ _)) args
  | Prim (Heap _) args =>
    shortest cs (map (lifetime_of cs) args)
  | Prim _ _ => Forever
  | Project_value_slot _ closure
  | Project_value_slot_from _ _ closure
  | Project_function_slot _ _ closure =>
    lifetime_of cs closure
  | Set_of_closures _ value_slots | Closure_of_set _ _ value_slots =>
    shortest cs (map (fun '(_, s) => lifetime_of cs s) value_slots)
  end.

(* ======================================================================== *)
(* A jump                                                                   *)
(* ======================================================================== *)

(* The regions given at the parameters that are regions: each must be
   the variable of a region that is open. *)
Fixpoint region_args (cs : check_state) (params : list variable)
    (kinds : list flambda_kind) (args : list (option simple * lifetime))
    : option (list (variable * variable)) :=
  match params, kinds, args with
  | [], [], [] => Some []
  | p :: params, kind :: kinds, (s, _) :: args =>
    let* rest := region_args cs params kinds args in
    if kind_eqb kind Region
    then
      match s with
      | Some (Var r) =>
        if among r (cs_open cs) then Some ((p, r) :: rest) else None
      | _ => None
      end
    else Some rest
  | _, _, _ => None
  end.

Definition region_at_jump (given : list (variable * variable)) (r : variable)
    : variable :=
  match lookup r given with
  | Some r' => r'
  | None => r
  end.

(* The nearest of the regions [named] at or below [r] among the open
   ones: what a value of [r] lives as long as, once only [named] can
   be ended by name. [None]: [r] is not open. [Some None]: no region
   of [named] is at or below it, and the value lives as long as the
   code. *)
Definition nearest_named (named open : list variable) (r : variable)
    : option (option variable) :=
  if among r open
  then Some (find (fun x => among x named) (r :: regions_below r open))
  else None.

(* Whether a value of the lifetime [given], at a jump from where
   [open] are open to a handler that is entered with [named] of them,
   lives as long as the handler takes its parameter to: [wanted], in
   the names of the jump. *)
Definition outlives (named open : list variable) (given wanted : lifetime)
    : bool :=
  match given, wanted with
  | _, Gone => true
  | Forever, _ => true
  | Gone, _ => false
  | (Of_caller | Of_region _), Forever => false
  | Of_caller, Of_caller => true
  | Of_caller, Of_region r' => among r' named
  | Of_region r, Of_caller =>
    match nearest_named named open r with
    | Some None => true
    | Some (Some _) | None => false
    end
  | Of_region r, Of_region r' =>
    among r' named
    && match nearest_named named open r with
       | Some None => true
       | Some (Some n) => among r' (down_to n open)
       | None => false
       end
  end.

Definition lifetime_at_jump (given : list (variable * variable))
    (l : lifetime) : lifetime :=
  match l with
  | Of_region r => Of_region (region_at_jump given r)
  | Forever | Of_caller | Gone => l
  end.

(* The arguments at the parameters that are not regions. *)
Fixpoint value_args (cs : check_state) (given : list (variable * variable))
    (named : list variable) (kinds : list flambda_kind)
    (args : list (option simple * lifetime)) (wanted : list lifetime)
    : bool :=
  match kinds, args, wanted with
  | [], [], [] => true
  | kind :: kinds, (_, l) :: args, w :: wanted =>
    (kind_eqb kind Region
     || (lifetime_usable cs l
         && outlives named (cs_open cs) l (lifetime_at_jump given w)))
    && value_args cs given named kinds args wanted
  | _, _, _ => false
  end.

(* A jump to [k] with arguments of the given lifetimes; the simple of
   an argument is read only at a parameter that is a region. A way out
   of the term takes any value that may be evaluated; the return
   continuation of code one that lives for ever, or any if the code
   may return a local value; its exception continuation only values
   that live for ever. *)
Definition jump (cs : check_state) (k : continuation)
    (args : list (option simple * lifetime)) : bool :=
  match lookup k (cs_conts cs) with
  | None => forallb (fun '(_, l) => lifetime_usable cs l) args
  | Some Unreachable => false
  | Some (To_return may_be_local) =>
    forallb
      (fun '(_, l) => lifetime_usable cs l && (may_be_local || is_forever l))
      args
  | Some To_raise => forallb (fun '(_, l) => is_forever l) args
  | Some (To_handler params kinds entry) =>
    match region_args cs params kinds args with
    | None => false
    | Some given =>
      let named := map (region_at_jump given) (he_open entry) in
      in_order named (cs_open cs)
      && value_args cs given named kinds args (he_params entry)
    end
  end.

(* What a call returns goes to [k]: to each of its parameters if it is
   a handler. *)
Definition jump_result (cs : check_state) (k : continuation)
    (result : lifetime) : bool :=
  match lookup k (cs_conts cs) with
  | Some (To_handler params _ _) =>
    jump cs k (map (fun _ => (None, result)) params)
  | _ => jump cs k [(None, result)]
  end.

Definition apply_cont_args (cs : check_state) (ac : apply_cont)
    : list (option simple * lifetime) :=
  map (fun s => (Some s, lifetime_of cs s)) (ac_args ac).

(* ======================================================================== *)
(* A call                                                                   *)
(* ======================================================================== *)

(* The number of arguments of a full call of code, counted by
   parameter: one, the tuple, of tupled code. *)
Definition full_arity (c : code) : nat :=
  if code_is_tupled c then 1 else length (code_params_arity c).

Definition max_full_arity (codes : list (code_id * code)) : nat :=
  fold_right (fun '(_, c) n => Nat.max n (full_arity c)) 0 codes.

(* The arguments against the parameters: one that may be local goes to
   a parameter that may be; one beyond the parameters goes to the
   callee that the code returns, which is not known. *)
Fixpoint args_fit (indirect_local : bool) (flags : list bool)
    (args : list lifetime) : bool :=
  match args with
  | [] => true
  | l :: args =>
    match flags with
    | [] => (is_forever l || indirect_local) && args_fit indirect_local [] args
    | flag :: flags =>
      (is_forever l || flag) && args_fit indirect_local flags args
    end
  end.

(* The greatest number of arguments of a full call of code a closure
   of which may be local. *)
Definition max_local_closure_arity (cert : certificate)
    (codes : list (code_id * code)) : nat :=
  fold_right
    (fun '(id, c) n =>
       match lookup id (cert_code cert) with
       | Some entry =>
         if ce_closure_local entry then Nat.max n (full_arity c) else n
       | None => n
       end)
    0 codes.

(* Whether a call with [count] arguments of a callee that is not known
   may be an application with too many: the full call takes fewer, and
   what it returns is called with the rest. It may if some code takes
   fewer; and a callee that is a partial application takes as many as
   its code still lacks, which may be one whenever some code takes two
   or more. *)
Definition later_call (codes : list (code_id * code)) (count : nat) : bool :=
  existsb (fun '(_, c) => full_arity c <? count) codes
  || ((1 <? max_full_arity codes) && (1 <? count)).

(* A call, with [count] arguments, of a callee that is not known: the
   callee of an indirect call, or what the full call inside an
   application with too many arguments returns. [None] if the
   certificate does not allow the call.

   The call is a chain: a full call of some code with the first
   arguments, then, if arguments remain, a call of what it returned
   with the rest, and so on. A link with fewer arguments than its
   callee takes makes a partial application, which keeps its callee
   and the arguments that it was given, and ends the chain. The first
   link may make one if some code takes more than [count] arguments; a
   later link, which has one argument or more, if some code takes two
   or more. A callee that is a partial application takes as many as
   its code still lacks, which is one at least: every partial
   application that a state holds was made by a call
   ([statics_no_partial]).

   What a full call returns lives for ever, or as long as the newest
   region if a callee that is not known may return a local value
   ([unknown]). A partial application lives as long as what it keeps:
   the first callee and all the arguments at the first link; at a
   later link what the link before returned and some of the arguments,
   which is no shorter than [unknown] and what the first would keep.

   If the certificate says that no partial application keeps a local
   value, the call is refused where a link could make one that does:
   one that keeps an argument that is not [Forever]; the first, of a
   callee that is not [Forever], which is then a local closure, if
   code of such a closure takes more than [count] arguments; a later
   one, of a local closure that the link before returned, if code of
   such a closure takes two arguments or more. Then every partial
   application is [Forever]. *)
Definition unknown_call (cert : certificate) (codes : list (code_id * code))
    (cs : check_state) (callee : lifetime) (args : list lifetime)
    (count : nat) : option lifetime :=
  let kept := shortest cs (callee :: args) in
  let unknown :=
    if cert_unknown_returns_local cert then newest_lifetime cs else Forever
  in
  let widest := max_full_arity codes in
  let widest_local := max_local_closure_arity cert codes in
  let later := later_call codes count in
  let may_be_partial := (count <? widest) || (later && (1 <? widest)) in
  if forallb is_forever args || cert_indirect_local cert
  then
    if cert_partial_local cert
    then
      Some
        (if may_be_partial then shorter (cs_open cs) unknown kept
         else unknown)
    else if
      (negb (forallb is_forever args) && may_be_partial)
      || (negb (is_forever callee) && (count <? widest_local))
      || (later && negb (is_forever unknown) && (1 <? widest_local))
    then None
    else Some unknown
  else None.

(* The lifetime of what a call returns; [None] if the certificate does
   not allow the call. Tupled code is given one tuple, and its
   parameters are the fields. A direct call with fewer arguments than
   a full call makes a partial application; with more, what the full
   call returns is called with the rest. *)
Definition call_result (cert : certificate) (codes : list (code_id * code))
    (cs : check_state) (ap : apply_expr) : option lifetime :=
  let callee := lifetime_of cs (ap_callee ap) in
  let args := map (lifetime_of cs) (ap_args ap) in
  let count := length (ap_args_arity ap) in
  match ap_call_kind ap with
  | Direct id =>
    let* c := lookup id codes in
    let* entry := lookup id (cert_code cert) in
    let flags :=
      if code_is_tupled c
      then [forallb (fun flag => flag) (ce_params_local entry)]
      else ce_params_local entry
    in
    if args_fit (cert_indirect_local cert) flags args
    then
      if count <? full_arity c
      then
        let kept := shortest cs (callee :: args) in
        if cert_partial_local cert then Some kept
        else if is_forever kept then Some Forever
        else None
      else
        let first :=
          if ce_returns_local entry then newest_lifetime cs else Forever
        in
        if count =? full_arity c then Some first
        else
          unknown_call cert codes cs first (skipn (length flags) args)
            (count - full_arity c)
    else None
  | Indirect => unknown_call cert codes cs callee args count
  end.

(* The code of the closures that a [Let] makes. *)
Definition closure_code (defining_expr : named) : list code_id :=
  match defining_expr with
  | Set_of_closures code _ => [code]
  | Closure_of_set _ function_decls _ => map snd function_decls
  | _ => []
  end.

Definition closure_may_be_local (cert : certificate) (id : code_id) : bool :=
  match lookup id (cert_code cert) with
  | Some entry => ce_closure_local entry
  | None => false
  end.

(* ======================================================================== *)
(* The check of an expression                                               *)
(* ======================================================================== *)

(* The lifetimes of the parameters of a handler, over the lifetimes
   where it is bound. A parameter that is a region may not be
   evaluated; one whose lifetime is a region that the handler is not
   entered with may not either. *)
Fixpoint bind_entry_params (he_open : list variable) (params : list variable)
    (kinds : list flambda_kind) (wanted : list lifetime)
    (env : list (variable * lifetime)) : list (variable * lifetime) :=
  match params, kinds, wanted with
  | p :: params, kind :: kinds, w :: wanted =>
    let l :=
      if kind_eqb kind Region then Gone
      else
        match w with
        | Of_region r => if among r he_open then w else Gone
        | Forever | Of_caller | Gone => w
        end
    in
    (p, l) :: bind_entry_params he_open params kinds wanted env
  | _, _, _ => env
  end.

Fixpoint region_params_of (params : list variable)
    (kinds : list flambda_kind) : list variable :=
  match params, kinds with
  | p :: params, kind :: kinds =>
    if kind_eqb kind Region
    then p :: region_params_of params kinds
    else region_params_of params kinds
  | _, _ => []
  end.

(* Whether an entry is one that a handler bound in the state [cs] may
   have: its parameters are new, as many as their kinds and their
   lifetimes; the regions it is entered with are parameters or were
   open where it is bound, each once. *)
Definition entry_fits (cs : check_state) (params : list variable)
    (kinds : list flambda_kind) (entry : handler_entry) : bool :=
  (length kinds =? length params)
  && (length (he_params entry) =? length params)
  && distinct params
  && forallb (not_a_region cs) params
  && distinct (he_open entry)
  && forallb
       (fun r =>
          among r (region_params_of params kinds) || among r (cs_open cs))
       (he_open entry).

(* The state in which a handler is checked: its entry, over the state
   where it is bound. *)
Definition handler_state (cs : check_state) (params : list variable)
    (kinds : list flambda_kind) (entry : handler_entry) : check_state :=
  Check_state
    (bind_entry_params (he_open entry) params kinds (he_params entry)
       (cs_env cs))
    (he_open entry)
    (region_params_of params kinds ++ cs_regions cs)
    (cs_conts cs).

Definition cont_entry_of (cert : certificate) (k : continuation)
    (h : continuation_handler) : cont_entry :=
  match h, lookup k (cert_handlers cert) with
  | Handler params metadata _, Some entry =>
    To_handler params (hm_params_kinds metadata) entry
  | _, None => Unreachable
  end.

Definition with_conts (cs : check_state)
    (conts : list (continuation * cont_entry)) : check_state :=
  Check_state (cs_env cs) (cs_open cs) (cs_regions cs) (conts ++ cs_conts cs).

(* The state in which the body of code is checked: no region is open;
   the closure and the parameters live for ever, or as long as the
   code where the entry says that they may be local; the two
   continuations of the code. *)
Definition code_state (entry : code_entry) (params : list variable)
    (my_closure : variable) (return_continuation exn_continuation
      : continuation) : check_state :=
  Check_state
    ((my_closure, if ce_closure_local entry then Of_caller else Forever)
     :: combine params
          (map (fun flag : bool => if flag then Of_caller else Forever)
             (ce_params_local entry)))
    [] []
    [(return_continuation, To_return (ce_returns_local entry));
     (exn_continuation, To_raise)].

Section Check.

Variable cert : certificate.

(* All the code that the machine can run: that of the static
   environment and that which the term binds. *)
Variable codes : list (code_id * code).

Fixpoint check_expr (cs : check_state) (e : expr) {struct e} : bool :=
  match e with
  (* The machine has no step for a primitive of regions that is not as
     the conversion makes it: the variable of a region is of the kind
     [Region], what an end binds of the kind [Value]. *)
  | Let x kind (Prim (Region_primitive p) args) body =>
    match p, args with
    | (Begin_region | Begin_try_region), [] =>
      kind_eqb kind Region
      && not_a_region cs x
      && check_expr
           (Check_state ((x, Gone) :: cs_env cs) (x :: cs_open cs)
              (x :: cs_regions cs) (cs_conts cs))
           body
    | (End_region | End_try_region), [Var r] =>
      (* Decision D47. The regions above the one that is ended end
         with it. *)
      kind_eqb kind Value
      && not_a_region cs x
      && among r (cs_open cs)
      && check_expr
           (Check_state ((x, Forever) :: cs_env cs)
              (regions_below r (cs_open cs)) (cs_regions cs) (cs_conts cs))
           body
    | _, _ => false
    end
  | Let x kind defining_expr body =>
    let simples := simples_of_named defining_expr in
    not_a_region cs x
    && forallb (check_usable cs) simples
    && match marker_of x body with
       | Marker y region _ =>
         (* A local allocation: of what allocates, of the kind
            [Value], or the machine has no step. Decision D47 of the
            region it names. A closure that is local is of code whose
            closures may be. What follows the marker is the body of
            the second [Let]: it is read from [body] so that the
            recursion is on a part of the expression. *)
         is_allocation defining_expr
         && kind_eqb kind Value
         && not_a_region cs y
         && match region with
            | None => true
            | Some (Var r) => among r (cs_open cs)
            | Some (Symbol _ | Const _) => false
            end
         && forallb (closure_may_be_local cert) (closure_code defining_expr)
         && match body with
            | Let _ _ _ body' =>
              check_expr
                (bind_lifetime y (newest_lifetime cs)
                   (bind_lifetime x (newest_lifetime cs) cs))
                body'
            | _ => false
            end
       | Bad_marker => false
       | No_marker =>
         if is_allocation defining_expr
         then
           (* Decision D46. *)
           forallb (fun s => is_forever (lifetime_of cs s)) simples
           && check_expr (bind_lifetime x Forever cs) body
         else
           check_expr (bind_lifetime x (named_lifetime cs defining_expr) cs)
             body
       end
  | Let_static bindings body =>
    (* Decision D46. *)
    forallb (fun s => is_forever (lifetime_of cs s))
      (flat_map simples_of_binding bindings)
    && forallb check_binding bindings
    && check_expr cs body
  | Let_cont_non_recursive k h body =>
    check_handler cs k h
    && check_expr (with_conts cs [(k, cont_entry_of cert k h)]) body
  | Let_cont_recursive handlers body =>
    let cs' :=
      with_conts cs (map (fun '(k, h) => (k, cont_entry_of cert k h)) handlers)
    in
    forallb (fun '(k, h) => check_handler cs' k h) handlers
    && check_expr cs' body
  | Apply ap =>
    (* What the call returns goes to the return continuation; the
       exception continuation is given the exception, which lives for
       ever, and the extra arguments. *)
    check_usable cs (ap_callee ap)
    && forallb (check_usable cs) (ap_args ap)
    && forallb (check_usable cs) (exn_extra_simples ap)
    && match call_result cert codes cs ap with
       | None => false
       | Some result =>
         jump_result cs (ap_continuation ap) result
         && jump cs (ap_exn_continuation ap)
              ((None, Forever)
               :: map (fun s => (Some s, lifetime_of cs s))
                    (exn_extra_simples ap))
       end
  | Apply_cont ac => jump cs (ac_continuation ac) (apply_cont_args cs ac)
  | Switch scrutinee arms =>
    check_usable cs scrutinee
    && forallb
         (fun '(_, ac) => jump cs (ac_continuation ac) (apply_cont_args cs ac))
         arms
  | Invalid => true
  end
(* A handler with no entry is not checked: nothing may jump to it. *)
with check_handler (cs : check_state) (k : continuation)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    match lookup k (cert_handlers cert) with
    | None => true
    | Some entry =>
      let kinds := hm_params_kinds metadata in
      entry_fits cs params kinds entry
      && check_expr (handler_state cs params kinds entry) handler
    end
  end
with check_binding (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code id c => check_code id c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
(* Code against its entry. The code has a parameter for each variable
   that its arity stands for, and the entry a flag for each: a call
   with fewer arguments than the code takes keeps them in a partial
   application without binding the parameters, so the machine does not
   test this there. If a callee that is not known may be given a
   local value, every parameter of every code may be; if none may
   return one, no code does. The parameters of tupled code are the
   fields of one tuple: all may be local or none. *)
with check_code (id : code_id) (c : code) {struct c} : bool :=
  match c with
  | Code params my_closure return_continuation exn_continuation metadata
      body =>
    match lookup id (cert_code cert) with
    | None => false
    | Some entry =>
      let flags := ce_params_local entry in
      (length flags =? length params)
      && (length params =? cardinal_unarized (cm_params_arity metadata))
      && distinct (my_closure :: params)
      && negb (Pos.eqb return_continuation exn_continuation)
      && (negb (cert_indirect_local cert) || forallb (fun flag => flag) flags)
      && (negb (cm_is_tupled metadata)
          || forallb (fun flag => flag) flags
          || forallb negb flags)
      && (negb (ce_returns_local entry) || cert_unknown_returns_local cert)
      && check_expr
           (code_state entry params my_closure return_continuation
              exn_continuation)
           body
    end
  end.

End Check.

(* ======================================================================== *)
(* The check of a term                                                      *)
(* ======================================================================== *)

(* The code that an expression binds. *)
Fixpoint code_of_expr (e : expr) : list (code_id * code) :=
  match e with
  | Let _ _ _ body => code_of_expr body
  | Let_static bindings body =>
    flat_map code_of_binding bindings ++ code_of_expr body
  | Let_cont_non_recursive _ h body => code_of_handler h ++ code_of_expr body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => code_of_handler h) handlers ++ code_of_expr body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with code_of_handler (h : continuation_handler) : list (code_id * code) :=
  match h with
  | Handler _ _ handler => code_of_expr handler
  end
with code_of_binding (binding : static_binding) : list (code_id * code) :=
  match binding with
  | Bind_code id c => (id, c) :: code_of_code c
  | Bind_deleted_code _ | Bind_symbol _ _ => []
  end
with code_of_code (c : code) : list (code_id * code) :=
  match c with
  | Code _ _ _ _ _ body => code_of_expr body
  end.

(* All the code that a run of a term in a static environment can run:
   that of the static environment and that which the term binds. The
   code that the body of code of the static environment binds is not
   there, and need not be: a call of code whose body binds something
   static has no step (FlambdaSemanticsSpec.call_code, the test of
   [binds_statics]), and neither has a recursive [Let_cont] one of
   whose handlers does. For the same reason the code that the body of
   code of the term binds, which is in the list, is never bound. If
   that rule of the machine changes, this list must. *)
Definition all_code (statics : static_env) (e : expr)
    : list (code_id * code) :=
  static_code statics ++ code_of_expr e.

Definition initial_check_state : check_state := Check_state [] [] [] [].

(* A value holds no partial application, however deep. *)
Fixpoint value_no_partial (v : value) : bool :=
  match v with
  | Value_const _ | Value_boxed_number _ | Value_location _
  | Value_symbol _ =>
    true
  | Value_closure _ value_slots | Value_closure_of_set _ _ value_slots =>
    forallb (fun '(_, v) => value_no_partial v) value_slots
  | Value_block _ fields => forallb value_no_partial fields
  | Value_immutable_array elements => forallb value_no_partial elements
  | Value_partial_application _ _ _ => false
  end.

Definition allocated_no_partial (allocated : statically_allocated_value)
    : bool :=
  match allocated with
  | Static_closure _ value_slots | Static_closure_of_set _ _ _ value_slots =>
    forallb (fun '(_, v) => value_no_partial v) value_slots
  | Static_block _ fields => forallb value_no_partial fields
  | Static_boxed_number contents => value_no_partial contents
  | Static_string _ | Static_imported _ => true
  | Static_immutable_array elements => forallb value_no_partial elements
  end.

(* The static environment that a term is run in holds no partial
   application. A partial application that a call makes keeps fewer
   arguments than its code takes, and [unknown_call] counts on that:
   one that kept as many, called with one argument more, would be an
   application with too many. A static environment is any list of
   bindings, so the one that the run starts in is asked to hold none;
   that of a unit, made of its imports before anything has run, holds
   none. This is not asked of what a [Let_static] of the term binds
   later, which may hold a partial application that a call made. *)
Definition statics_no_partial (statics : static_env) : bool :=
  forallb (fun '(_, allocated) => allocated_no_partial allocated)
    (static_values statics).

(* The check: no code is bound twice; the static environment holds no
   partial application; every code of the static environment is as its
   entry says; the term is. *)
Definition region_check (cert : certificate) (statics : static_env)
    (e : expr) : bool :=
  let codes := all_code statics e in
  distinct (map fst codes)
  && statics_no_partial statics
  && forallb (fun '(id, c) => check_code cert codes id c) (static_code statics)
  && check_expr cert codes initial_check_state e.

(* ======================================================================== *)
(* Statements                                                               *)
(* ======================================================================== *)

(* A run on the machine of regions that never stops without a step. *)
Definition region_safe (rs : region_state) : Prop :=
  ∀ (fuel : nat) (fault : region_fault),
    region_run fuel rs <> Region_undefined fault.

(* The regions of a term add no fault to it: where its run on the
   machine of regions stops without a step, the run of the term with
   its regions dropped goes wrong too. Nothing is said of what goes
   wrong without regions. *)
Definition regions_add_no_fault (e : expr) (statics : static_env)
    (base : symbol) : Prop :=
  ∀ (fuel : nat) (fault : region_fault),
    region_run fuel (initial_region_state e statics base)
    = Region_undefined fault ->
    behaves (erased_initial_state e statics) Goes_wrong.

(* The check is sound: the regions of a term that passes it, with any
   certificate, add no fault to the term. *)
Definition region_check_sound_statement : Prop :=
  ∀ (cert : certificate) (statics : static_env) (base : symbol) (e : expr),
    region_check cert statics e = true ->
    region_names_fresh base e statics = true ->
    erasable e statics = true ->
    regions_add_no_fault e statics base.

(* What it gives with the validation of a run of Simplify, which
   proves [refines] of the two terms with their regions dropped. If
   the term before, with its regions dropped, does not go wrong, and
   the regions of the term after add no fault to it, the term after,
   with its regions, never stops without a step. Nothing is asked of
   the regions of the term before. *)
Definition checked_after_safe_statement : Prop :=
  ∀ (statics : static_env) (base : symbol) (before after : expr),
    refines (erased_initial_state before statics)
      (erased_initial_state after statics) ->
    ~ behaves (erased_initial_state before statics) Goes_wrong ->
    regions_add_no_fault after statics base ->
    region_safe (initial_region_state after statics base).

(* And then the two runs compute the same: an exit of the term before
   with its regions is an exit of the term after with its regions,
   through the same continuation, with values that are erased to the
   same values where they may be evaluated. *)
Definition checked_exits_agree_statement : Prop :=
  ∀ (statics : static_env) (base : symbol) (before after : expr),
    region_names_fresh base before statics = true ->
    region_names_fresh base after statics = true ->
    erasable before statics = true ->
    erasable after statics = true ->
    refines (erased_initial_state before statics)
      (erased_initial_state after statics) ->
    ~ behaves (erased_initial_state before statics) Goes_wrong ->
    regions_add_no_fault after statics base ->
    ∀ (fuel : nat) (k : continuation) (vs : list value) (rs : region_state),
      region_run fuel (initial_region_state before statics base)
      = Region_exit k vs rs ->
      ∃ (fuel' : nat) (ws : list value) (rs' : region_state)
        (us : list value),
        region_run fuel' (initial_region_state after statics base)
        = Region_exit k ws rs'
        ∧ Forall2 (value_erased_if_usable rs) vs us
        ∧ Forall2 (value_erased_if_usable rs') ws us.
