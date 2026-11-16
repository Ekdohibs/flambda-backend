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


(* Unboxing a parameter of a continuation that is read under a test,
   with a constant given where there is nothing to read: a
   normalisation that refines. It is what the unboxing of a variant
   needs.

   This file is to be reviewed: [unbox_variant_params] and what it is
   made of, and the statement. The validation runs the pass in the
   first phase of its normalisation of the term before
   (FlambdaValidateSpec.normalise_expr_params).

   FlambdaUnboxParamsSpec.v makes a parameter of the [Let] that a
   handler begins with, where that [Let] reads a box which is a
   parameter and every application gives a box that the term shows.
   That is not enough for a variant. The handler of a join of [None]
   and [Some x] begins with [Is_int] of its parameter, which that pass
   unboxes, and then goes by a [Switch] to a continuation that begins
   with the load of the field; and one application gives the constant
   [None], which has no field. Simplify unboxes such a parameter
   ([Unboxing_epa.compute_extra_args_for_variant]): the application
   that gives a block gives its field, and the one that gives a
   constant constructor gives a constant in its place, [poison] of the
   kind of the field ([access_kind_and_poison_const]); for the tag, 0
   ([Tag.Scannable.zero]); for the constant constructor of an
   application that gives a block, 0 as well ([extra_arg_for_ctor]).

   [unbox_variant_params] is made as the other pass is, with the
   differences below. It is not the other pass and more: it takes the
   reads in an order of its own and stops at the first that it cannot
   make a parameter, so it may leave a handler of which the other
   pass makes the first read a parameter (below, at what stops it).

   - The [Let] need not be the first thing in the handler. It is the
     first, going down the handler, that reads a box which is a
     parameter ([first_read]): through the bodies of [Let]s, and
     through the bodies and then the handlers of non-recursive
     [Let_cont]s, as long as no binder on the way binds the parameter
     read or the variable of the [Let]. The variable must not be free
     in the handler ([var_free_b]): it becomes a parameter, in scope
     in all of the handler. The [Let] is removed where it is and its
     variable is the new parameter, the last.

   - The reads are taken in an order ([next_read]): first a [Get_tag],
     then an [Untag_immediate], then an [Is_int], then the loads and
     the unboxings, each sort from the top of the handler down. That
     is the order of Simplify's new parameters for a variant: the tag,
     the constant constructor, whether the value is an immediate, then
     the fields ([Unboxing_epa.add_extra_params_and_args], each
     parameter put before those already there, the fields first). The
     body of a [Let_cont] is gone into before its handler: the front
     end puts the continuation of the first arm of a match outermost,
     and Simplify's parameters for the fields are those of the last
     tag first. Below a load, a load of the same box at a greater
     field is taken before it ([first_read], with [loads_above]):
     Simplify's parameters for the fields of a block are those of the
     last field first, and the front end loads the fields of a
     constructor from the first.

   - Where the [Let] is not the first thing in the handler
     ([extend_args_certain]), an application gets as its new argument:
     a constant of the kind of the [Let] in place of what the read
     would give ([dummy_of], as Simplify gives: [poison]; or, at the
     kind of naked immediates, 0 for [Get_tag], for [Is_int], and
     for the [Untag_immediate] of a box of which the handler also
     reads [Is_int], which is the constant constructor of a variant
     ([extra_arg_for_ctor], [extra_arg_for_is_int]); a box that is
     untagged and not so tested is a number, for which Simplify gives
     [poison], [Unboxers.Immediate]), if the application does not
     come to the read ([never_reads], below);
     otherwise the constant that the term shows the primitive to give,
     if that is of the kind of the [Let]; or the field of a block that
     a [Let] in scope made, for a load at the kind of values; or the
     number in a box that a [Let] in scope made, for the unboxing of
     such a box at the kind of the number; or the number that a [Let]
     in scope tagged, for the untagging at the kind of naked
     immediates ([certain_projection]); or,
     where the primitive has no value on the argument for the
     parameter read, that constant again. The
     primitive is known to have no value ([no_value_on]) where it is a
     load or [Get_tag] and the argument is a constant, and where it is
     [Untag_immediate] and the argument is a variable that a [Let] in
     scope made a block, or a symbol, which is never an immediate.
     Where the [Let] is the first thing in the
     handler ([head_read]) the argument is the one of the other pass
     ([extend_args]).

   - A load is not made a parameter where the applications that may
     come to it give blocks of more than one tag
     ([loads_of_two_tags]), and the pass stops there. For a variant
     Simplify makes the parameters for the fields tag by tag
     ([fields_by_tag] of [Unboxing_types.decision]) and gives the
     parameter unboxed the type of a variant that has, at each tag,
     those of that tag ([Build_unboxing_denv.denv_of_decision]); a
     load in the handler is replaced by one of them only where that
     type gives one ([Simplify_unary_primitive]'s
     [simplify_immutable_block_load]), which it does where the tag is
     known. Where blocks of two tags come to a load, no [Switch] on
     the tag having been gone through, the load stays, its box stays
     a parameter and the applications give the blocks as they did.
     The tags are those of the arguments that a [Let] in scope or a
     [Let_static] made a block (FlambdaBlockLoadSpec.known_block_of),
     in the applications that the term does not show to be kept from
     the load ([never_reads], with the tests of the same box, as for
     the constant above); an argument that is not such a block has
     no tag here.

   - Where a read is made a parameter, the other [Let]s of the handler
     that read the same parameter by the same primitive become [Let]s
     of the variable of the read ([with_reads_shared], by
     [share_reads]): Simplify has one parameter for what a handler
     reads in several places. Below such a [Let], a unary primitive
     applied to its variable and a [Switch] on it are of the variable
     of the read; so a read of what was read twice is a read of the
     new parameter, and a test that was made twice is one test for
     [never_reads]. The rounds of the validation then follow those
     [Let]s and remove them. Loads are shared only where the tags
     that are known for them, taken together, do not differ
     ([shares_reads], which is [loads_of_two_tags] of the read with
     the others shared): Simplify's parameters for the fields of a
     variant are tag by tag, so the loads of one field under two
     tags are two parameters. The tags are those above: of the
     arguments that the term shows to be blocks that were made, in
     every application but those that it shows to come to none of
     the loads. An argument that is not such a block has no tag
     here, so loads are shared also where no tag is known, or one,
     and blocks of other tags may then come to them. The test
     chooses what is shared and nothing more: the refinement does
     not rest on it. Where the loads are not shared the read is
     taken alone.

   An application does not come to a read ([never_reads]) where the
   term shows it by the constants that the application gives. The
   parameters that it gives a constant naked immediate are known
   ([known_immediates]): whether the value is an immediate and its
   tag, once those are parameters. Of them, only the tests of the box
   read are used ([tests_of]): the parameters that the pass itself
   made from [Get_tag], [Is_int] or [Untag_immediate] of the
   parameter read, or of a parameter that the parameter read was
   itself read from. Simplify gives [poison] for the fields of a
   block of another tag and for everything under a value that is not
   there, and the field itself wherever the block is there, whatever
   else the application shows of the path that the handler takes: a
   pair [(0, false)] given to a handler that reads the first field
   only where the second is [true] is given 0 for it. In the handler
   each [Switch] on one of them is replaced by the arm that it goes
   to ([take_arms]), and
   the continuations that nothing then applies are removed
   (FlambdaUnusedContsSpec.remove_unused_conts); if what is left of
   the handler is what is left of the handler without the read, up to
   bound names (FlambdaCompareSpec.expr_eqb), the read was in a
   continuation that this application does not reach. That is how the
   application that gives a block of one tag gives [poison] for the
   fields of another: the front end's load says nothing of the tag,
   so it has a value on any block that has the field, and only the
   [Switch] on the tag keeps it from being read.

   Why that constant may be given. The new parameter
   is read only below the place of the [Let] that was removed. In the
   term before, a run that comes to that place evaluates the
   primitive on the parameter read, which no binder on the way has
   bound again, so it is the value that the application gave. The
   constant is given in two cases. In the first, the primitive has no
   value on the argument of the application: such a run of the term
   before is undefined when it comes to the place of the [Let], and a
   refinement asks nothing of the term after; a run that does not
   come to that place does not look at the new parameter. In the
   second, the application does not come to the read: the handler and
   the handler without the read are equivalent under the constants
   that the application gives, since each is equivalent to what is
   left of it (the two rewrites are equivalences where a variable has
   the number it is known to have) and the two that are left are the
   same up to bound names. So nothing is claimed of what a primitive
   does with [poison], and the statement has no premise: the constant
   could be any constant of the kind. It has to be of that kind: a
   handler is applied only to values of the kinds of its parameters
   (FlambdaSemanticsSpec.apply_cont_value), and the run of the term
   before, which may never come to the [Let], does not test anything
   there. That is also why, under a test, the argument given must be
   known to be of the kind of the [Let] whether or not the primitive
   has a value on the box: the field of a block that [Make_block]
   made is of the kind of values, the number in a box that
   [Box_number] made is of the kind of the box, and the number that
   [Tag_immediate] tagged is a naked immediate, as each primitive
   tested when it made the block, the box or the tagged immediate; of
   a field of a static block nothing is tested, so a symbol is not
   taken as a known block here.

   What stops it for a parameter: what stops the other pass; a load
   to which blocks of more than one tag may come; and,
   under a test, an application that may come to the read and whose
   argument is neither known in one of these ways nor one on which
   the primitive has no value. It stops at the first read that it
   cannot make a parameter, and does not look for another. A load
   that is taken before the load that the handler begins with is not
   the first thing in the handler, and is given arguments that are
   certain. So where
   a handler begins with the load of a field of one parameter and
   then reads the tag of another, and an application gives a block
   that is known for the first and nothing that is known for the
   second, the other pass makes the load a parameter and this one
   makes none: the tag comes first in its order.

   The result refines the term
   ([unbox_variant_params_refines_statement]), under all
   environments; so the pass is for the term before Simplify. It
   makes no name. FlambdaUnboxVariantsRenameSpec.v has the statements
   that it commutes with a renaming and that it keeps scope.

   The proofs are in FlambdaUnboxVariantsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel FlambdaMapSpec
  FlambdaPrimitiveModel FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaNormaliseSpec FlambdaBlockLoadSpec FlambdaUnboxSpec
  FlambdaUnusedContsSpec FlambdaCompareSpec FlambdaUnboxParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The argument for a parameter that is read under a test                   *)
(* ======================================================================== *)

(* The constant given where there is nothing to read, of the kind of
   the [Let] that read: what Simplify gives. [variant] says whether
   the box read is also tested by [Is_int]: its untagging is then the
   constant constructor of a variant, and otherwise that of a
   number. *)
Definition dummy_of (variant : bool) (p : unary_primitive)
    (kind : flambda_kind) : const :=
  match p with
  | Get_tag | Is_int _ =>
    if kind_eqb kind (Naked_number Kind_naked_immediate)
    then Naked_immediate 0
    else Poison kind
  | Untag_immediate =>
    if variant && kind_eqb kind (Naked_number Kind_naked_immediate)
    then Naked_immediate 0
    else Poison kind
  | _ => Poison kind
  end.

(* Whether the term shows that the primitive has no value on the
   argument: a load or [Get_tag] of a constant, the untagging of what a
   [Let] in scope made a block, or of a symbol. *)
Definition no_value_on (known : facts) (p : unary_primitive) (arg : simple)
    : bool :=
  match p, arg with
  | (Block_load _ _ _ | Get_tag), Const _ => true
  | Untag_immediate, Var y => is_bound y (known_vars (facts_blocks known))
  | Untag_immediate, Symbol _ => true
  | _, _ => false
  end.

(* The field that a load not marked [Mutable] reads, at the kind of
   values, of a block that a [Let] in scope made. *)
Definition made_field (known : facts) (p : unary_primitive)
    (kind : flambda_kind) (arg : simple) : option simple :=
  match p, arg with
  | Block_load access (Immutable | Immutable_unique) field, Var y =>
    let* block := lookup y (known_vars (facts_blocks known)) in
    let* contents := field_loaded access field block in
    if kind_eqb kind Value then Some contents else None
  | _, _ => None
  end.

(* The number in a box that a [Let] in scope made, for the unboxing of
   a box of that sort, at the kind of the number; and the number that
   a [Let] in scope tagged, for the untagging, at the kind of naked
   immediates. *)
Definition made_number (known : facts) (p : unary_primitive)
    (kind : flambda_kind) (arg : simple) : option simple :=
  match p, arg with
  | Unbox_number boxed, Var y =>
    let* contents := unwrapped (facts_numbers known) (Boxed boxed) (Var y) in
    if kind_eqb kind (unboxed_kind boxed) then Some contents else None
  | Untag_immediate, Var y =>
    let* contents := unwrapped (facts_numbers known) Tagged (Var y) in
    if kind_eqb kind (Naked_number Kind_naked_immediate)
    then Some contents
    else None
  | _, _ => None
  end.

(* What the primitive gives of the argument, where the term shows it and
   shows that it is of the kind given: a constant of that kind, the
   field of a block that was made, or the number that was put in a box
   or tagged. *)
Definition certain_projection (known : facts) (p : unary_primitive)
    (kind : flambda_kind) (arg : simple) : option simple :=
  match projection_of known p arg with
  | Some (Const c) =>
    if kind_eqb (kind_of_const c) kind then Some (Const c) else None
  | Some (Var _ | Symbol _) | None =>
    match made_field known p kind arg with
    | Some contents => Some contents
    | None => made_number known p kind arg
    end
  end.

(* The parameters that an application gives a constant naked immediate,
   with that number. *)
Fixpoint known_immediates (params : list variable) (args : list simple)
    : list (variable * Z) :=
  match params, args with
  | param :: params, Const (Naked_immediate i) :: args =>
    (param, i) :: known_immediates params args
  | _ :: params, _ :: args => known_immediates params args
  | _, _ => []
  end.

(* What is known of the variables that are not bound again. *)
Definition forget_immediates (bound : list variable)
    (consts : list (variable * Z)) : list (variable * Z) :=
  filter (fun entry => negb (among (fst entry) bound)) consts.

(* The expression with each [Switch] on a variable whose number is
   known replaced by the arm that it goes to. *)
Fixpoint take_arms (consts : list (variable * Z)) (e : expr) : expr :=
  match e with
  | Let y kind d body =>
    Let y kind d (take_arms (forget_immediates [y] consts) body)
  | Let_static bindings body => Let_static bindings (take_arms consts body)
  | Let_cont_non_recursive k (Handler handler_params metadata handler) body =>
    Let_cont_non_recursive k
      (Handler handler_params metadata
         (take_arms (forget_immediates handler_params consts) handler))
      (take_arms consts body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (take_arms consts body)
  | Switch (Var t) arms =>
    match lookup t consts with
    | Some i =>
      match find_arm i arms with
      | Some ac => Apply_cont ac
      | None => e
      end
    | None => e
    end
  | Switch (Symbol _ | Const _) _ | Apply _ | Apply_cont _ | Invalid => e
  end.

(* Whether a handler and the handler without one of its reads are the
   same, up to bound names, once the arms that an application takes
   are taken and the continuations that nothing then applies are
   removed: the application does not come to the read. *)
Definition never_reads (consts : list (variable * Z))
    (handler without : expr) : bool :=
  expr_eqb [] []
    (remove_unused_conts (take_arms consts handler))
    (remove_unused_conts (take_arms consts without)).

(* What is known of the variables given, and of no other. *)
Definition only_of (tests : list variable) (consts : list (variable * Z))
    : list (variable * Z) :=
  filter (fun entry => among (fst entry) tests) consts.

(* The arguments of an application with one more, the last: the
   constant of [dummy_of], if the application does not come to the
   read, given the handler and the handler without the read, by the
   constants that it gives to the parameters [tests]; otherwise
   what [p] certainly gives of the argument at position [i]; or, if it
   has no value on it, the constant again. *)
Definition extend_args_certain (variant : bool)
    (tests params : list variable) (handler without : expr) (known : facts)
    (i : nat) (p : unary_primitive) (kind : flambda_kind)
    (args : list simple) : option (list simple) :=
  let* arg := nth_error args i in
  if never_reads (only_of tests (known_immediates params args)) handler
       without
  then Some (args ++ [Const (dummy_of variant p kind)])
  else
    match certain_projection known p kind arg with
    | Some projection => Some (args ++ [projection])
    | None =>
      if no_value_on known p arg
      then Some (args ++ [Const (dummy_of variant p kind)])
      else None
    end.

(* ======================================================================== *)
(* The first read of a parameter                                            *)
(* ======================================================================== *)

(* A [Let] that reads a box: its variable and its kind, the primitive,
   where the parameter read is, and the expression without the [Let]. *)
Record read : Type := Read {
  read_variable : variable;
  read_kind : flambda_kind;
  read_primitive : unary_primitive;
  read_position : nat;
  read_without : expr
}.

(* The sorts of reads, in the order in which they are made
   parameters. *)
Definition reads_tag (p : unary_primitive) : bool :=
  match p with
  | Get_tag => true
  | _ => false
  end.

Definition reads_constant_constructor (p : unary_primitive) : bool :=
  match p with
  | Untag_immediate => true
  | _ => false
  end.

Definition reads_is_int (p : unary_primitive) : bool :=
  match p with
  | Is_int _ => true
  | _ => false
  end.

Definition reads_contents (p : unary_primitive) : bool :=
  match p with
  | Block_load _ _ _ | Unbox_number _ => true
  | _ => false
  end.

Definition reads_any (p : unary_primitive) : bool := true.

(* Whether a [Let] of [f] to [d] reads, by a primitive that is wanted,
   a box which is one of the parameters, neither it nor [f] being among
   the variables bound on the way: the primitive, and where the
   parameter is. *)
Definition param_read (wanted : unary_primitive -> bool)
    (params bound : list variable) (f : variable) (d : named)
    : option (unary_primitive * nat) :=
  match d with
  | Prim (Unary p) [Var x] =>
    if wanted p && reads_a_box p && negb (among x bound)
       && negb (among f bound)
    then option_map (fun i => (p, i)) (position x params)
    else None
  | _ => None
  end.

(* The [Let] that the expression begins with, if it reads a
   parameter. *)
Definition head_read (params : list variable) (e : expr) : option read :=
  match e with
  | Let f kind d rest =>
    let* '(p, i) := param_read reads_any params [] f d in
    Some (Read f kind p i rest)
  | _ => None
  end.

(* The variable that a load reads, and the field. *)
Definition load_read (d : named) : option (variable * Z) :=
  match d with
  | Prim (Unary (Block_load _ _ field)) [Var x] => Some (x, field)
  | _ => None
  end.

(* Whether a primitive is a load of a field above the one given. *)
Definition loads_above (field : Z) (p : unary_primitive) : bool :=
  match p with
  | Block_load _ _ field' => (field <? field')%Z
  | _ => false
  end.

(* The first [Let] that reads a parameter by a primitive that is
   wanted, going down the expression through the bodies of [Let]s and
   through the bodies, then the handlers, of non-recursive [Let_cont]s;
   [bound] is what the binders passed on the way bind. Where that
   [Let] is a load, the first load of the same parameter at a greater
   field that is found in the same way below it is taken in its
   place, and so on down. *)
Fixpoint first_read (wanted : unary_primitive -> bool)
    (params bound : list variable) (e : expr) : option read :=
  match e with
  | Let y kind d body =>
    match param_read wanted params bound y d with
    | Some (p, i) =>
      match
        (let* '(x, field) := load_read d in
         first_read (loads_above field) [x] (y :: bound) body)
      with
      | Some r =>
        Some
          (Read (read_variable r) (read_kind r) (read_primitive r) i
             (Let y kind d (read_without r)))
      | None => Some (Read y kind p i body)
      end
    | None =>
      let* r := first_read wanted params (y :: bound) body in
      Some
        (Read (read_variable r) (read_kind r) (read_primitive r)
           (read_position r) (Let y kind d (read_without r)))
    end
  | Let_cont_non_recursive k (Handler handler_params metadata handler) body =>
    match first_read wanted params bound body with
    | Some r =>
      Some
        (Read (read_variable r) (read_kind r) (read_primitive r)
           (read_position r)
           (Let_cont_non_recursive k
              (Handler handler_params metadata handler) (read_without r)))
    | None =>
      let* r :=
        first_read wanted params (handler_params ++ bound) handler
      in
      Some
        (Read (read_variable r) (read_kind r) (read_primitive r)
           (read_position r)
           (Let_cont_non_recursive k
              (Handler handler_params metadata (read_without r)) body))
    end
  | Let_static _ _ | Let_cont_recursive _ _ | Apply _ | Apply_cont _
  | Switch _ _ | Invalid =>
    None
  end.

(* The read that is made a parameter next: of the first sort that the
   handler has a read of. *)
Definition next_read (params : list variable) (handler : expr)
    : option read :=
  match first_read reads_tag params [] handler with
  | Some r => Some r
  | None =>
    match first_read reads_constant_constructor params [] handler with
    | Some r => Some r
    | None =>
      match first_read reads_is_int params [] handler with
      | Some r => Some r
      | None => first_read reads_contents params [] handler
      end
    end
  end.

(* How many [Let]s [first_read] may come to: more reads than that are
   not made parameters. *)
Fixpoint lets_in (e : expr) : nat :=
  match e with
  | Let _ _ _ body => S (lets_in body)
  | Let_cont_non_recursive _ (Handler _ _ handler) body =>
    lets_in handler + lets_in body
  | Let_static _ _ | Let_cont_recursive _ _ | Apply _ | Apply_cont _
  | Switch _ _ | Invalid =>
    0
  end.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* The [Let_cont] of [k] with the read made a parameter, if the handler
   may take its variable, that variable is not free in the handler, and
   every application in [body] can be given its argument by [extend]. *)
Definition unbox_read
    (extend : facts -> list simple -> option (list simple)) (known : facts)
    (k : continuation) (params : list variable) (metadata : handler_metadata)
    (handler body : expr) (r : read)
    : option (list variable * handler_metadata * expr * expr) :=
  if may_take params metadata (read_variable r)
     && negb (var_free_b (read_variable r) handler)
  then
    let* body := extend_applications_by extend known k body in
    Some
      (params ++ [read_variable r], with_one_more metadata (read_kind r),
       read_without r, body)
  else None.

(* Whether the handler reads [Is_int] of the parameter at position
   [i]. *)
Definition tested_is_int (params : list variable) (i : nat) (handler : expr)
    : bool :=
  match nth_error params i with
  | Some param => is_some (first_read reads_is_int [param] [] handler)
  | None => false
  end.

(* The reads that are tests of a box. *)
Definition reads_test (p : unary_primitive) : bool :=
  reads_tag p || reads_constant_constructor p || reads_is_int p.

(* The positions of the parameter at position [i] and of the
   parameters that it was read from, one after the other, given for
   each parameter that the pass made the positions of its own. *)
Fixpoint boxes_of (i : nat) (above : list (nat * list nat)) : list nat :=
  match above with
  | [] => [i]
  | (j, boxes) :: above => if Nat.eqb i j then boxes else boxes_of i above
  end.

(* The tests of the parameter at position [i], and of the parameters
   that it was read from: of the parameters that the pass made from a
   test, each with the position that it read, those that read one of
   those positions. *)
Definition tests_of (i : nat) (made : list (variable * nat))
    (above : list (nat * list nat)) : list variable :=
  map fst
    (filter (fun entry => existsb (Nat.eqb (snd entry)) (boxes_of i above))
       made).

(* The numbers that [tag_of] gives for the applications of [k] in an
   expression, one after the other, with what is known at each: the
   applications are those that
   FlambdaUnboxParamsSpec.extend_applications_by extends, and what is
   known at each is what it is there. *)
Definition tags_at_apply_cont (tag_of : facts -> list simple -> list Z)
    (known : facts) (k : continuation) (ac : apply_cont) : list Z :=
  if Pos.eqb (ac_continuation ac) k then tag_of known (ac_args ac) else [].

Fixpoint tags_given (tag_of : facts -> list simple -> list Z) (known : facts)
    (k : continuation) (e : expr) : list Z :=
  match e with
  | Let x _ d body => tags_given tag_of (facts_under_let known x d) k body
  | Let_static bindings body =>
    tags_given tag_of (facts_under_statics known bindings) k body
  | Let_cont_non_recursive k' h body =>
    tags_given_in_handler tag_of known k h
    ++ (if Pos.eqb k' k then [] else tags_given tag_of known k body)
  | Let_cont_recursive handlers body =>
    if is_bound k handlers then []
    else
      flat_map (fun '(_, h) => tags_given_in_handler tag_of known k h)
        handlers
      ++ tags_given tag_of known k body
  | Apply_cont ac => tags_at_apply_cont tag_of known k ac
  | Switch _ arms =>
    flat_map (fun '(_, ac) => tags_at_apply_cont tag_of known k ac) arms
  | Apply _ | Invalid => []
  end
with tags_given_in_handler (tag_of : facts -> list simple -> list Z)
    (known : facts) (k : continuation) (h : continuation_handler) : list Z :=
  match h with
  | Handler params _ handler =>
    tags_given tag_of (facts_under_params known params) k handler
  end.

(* The tag of the block that an application gives for the parameter at
   position [i], where the term shows the argument to be a block that
   was made and does not show the application to be kept from the
   read, given the handler and the handler without the read, by the
   constants that it gives to the parameters [tests]: that tag alone,
   and otherwise nothing. *)
Definition tag_read (tests params : list variable) (handler without : expr)
    (i : nat) (known : facts) (args : list simple) : list Z :=
  match nth_error args i with
  | Some arg =>
    if never_reads (only_of tests (known_immediates params args)) handler
         without
    then []
    else
      match known_block_of (facts_blocks known) arg with
      | Some block => [fst block]
      | None => []
      end
  | None => []
  end.

(* Whether the numbers are all the same. *)
Definition all_same (tags : list Z) : bool :=
  match tags with
  | [] => true
  | tag :: tags => forallb (Z.eqb tag) tags
  end.

(* The reads that are loads of a field of a block. *)
Definition reads_field (p : unary_primitive) : bool :=
  match p with
  | Block_load _ _ _ => true
  | _ => false
  end.

(* Whether the read is a load to which blocks of more than one tag may
   come: the applications of [k] in [body] that may come to it give,
   for the parameter read, blocks that were made and that are not all
   of one tag. *)
Definition loads_of_two_tags (known : facts) (k : continuation)
    (tests params : list variable) (handler body : expr) (r : read) : bool :=
  reads_field (read_primitive r)
  && negb
       (all_same
          (tags_given
             (tag_read tests params handler (read_without r)
                (read_position r))
             known k body)).

(* Whether a [Let] is of a variable to the primitive [p] applied to
   [x]. *)
Definition reads_same (p : unary_primitive) (x : variable) (d : named)
    : bool :=
  match d with
  | Prim (Unary p') [Var x'] =>
    prim_eqb (Unary p') (Unary p) && Pos.eqb x' x
  | _ => false
  end.

(* A unary primitive applied to one of the variables [same], applied
   to [f] in its place. *)
Definition read_of_same (f : variable) (same : list variable) (d : named)
    : named :=
  match d with
  | Prim (Unary p') [Var z] =>
    if among z same then Prim (Unary p') [Var f] else d
  | _ => d
  end.

(* The expression with every other [Let] of a variable to the primitive
   [p] applied to [x] made a [Let] of that variable to [f], going down
   through the bodies of [Let]s and through the handlers and the bodies
   of non-recursive [Let_cont]s, as far as neither [x] nor [f] is
   bound again. [same] is the variables so bound that are in scope:
   each holds what [f] holds, so a unary primitive applied to one of
   them is applied to [f], and a [Switch] on one of them is on [f]. *)
Fixpoint share_reads (p : unary_primitive) (x f : variable)
    (same : list variable) (e : expr) : expr :=
  match e with
  | Let y kind d body =>
    let d := read_of_same f same d in
    let shared := reads_same p x d in
    Let y kind (if shared then Simple (Var f) else d)
      (if Pos.eqb y x || Pos.eqb y f
       then body
       else
         share_reads p x f
           (if shared then y :: same else without [y] same) body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    Let_cont_non_recursive k
      (Handler params metadata
         (if among x params || among f params
          then handler
          else share_reads p x f (without params same) handler))
      (share_reads p x f same body)
  | Switch (Var z) arms =>
    if among z same then Switch (Var f) arms else e
  | Switch (Symbol _ | Const _) _ | Let_static _ _ | Let_cont_recursive _ _
  | Apply _ | Apply_cont _ | Invalid =>
    e
  end.

(* The read with, if [share] says so, the other reads of the same
   parameter by the same primitive, in what is left of the handler,
   made [Let]s of its variable. *)
Definition with_reads_shared (share : bool) (params : list variable)
    (r : read) : read :=
  if share
  then
    match nth_error params (read_position r) with
    | Some x =>
      Read (read_variable r) (read_kind r) (read_primitive r)
        (read_position r)
        (share_reads (read_primitive r) x (read_variable r) []
           (read_without r))
    | None => r
    end
  else r.

(* Whether the other reads of the same parameter by the same primitive
   are shared with the read: always, but for loads of which, taken
   together, [loads_of_two_tags] holds: of the applications that may
   come to one of them, two give blocks that were made, of different
   tags. So the loads are shared where no application is known to
   give such a block, and where those that are known are all of one
   tag; blocks of other tags may then come to them. *)
Definition shares_reads (known : facts) (k : continuation)
    (tests params : list variable) (handler body : expr) (r : read) : bool :=
  negb
    (loads_of_two_tags known k tests params handler body
       (with_reads_shared true params r)).

(* The [Let_cont] of [k], with the reads of its parameters made
   parameters one after the other, in the order of [next_read], as
   long as each can be: not a load to which blocks of more than one
   tag may come ([loads_of_two_tags]); if it is the [Let] that the
   handler begins
   with, as FlambdaUnboxParamsSpec.unbox_heads does it; otherwise with
   arguments that are certain. Each is taken with the other reads of
   the same parameter by the same primitive shared, where
   [shares_reads] says so. [made] is the tests made parameters so
   far, each with the position that it read, and [above], for each
   parameter made so far, its position and those of the parameters
   that it was read from. *)
Fixpoint unbox_reads (fuel : nat) (known : facts) (k : continuation)
    (made : list (variable * nat)) (above : list (nat * list nat))
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr) : expr :=
  let unchanged :=
    Let_cont_non_recursive k (Handler params metadata handler) body
  in
  match fuel with
  | 0 => unchanged
  | S fuel =>
    let certain (r : read) :=
      unbox_read
        (fun known =>
           extend_args_certain
             (tested_is_int params (read_position r) handler)
             (tests_of (read_position r) made above)
             params handler (read_without r) known
             (read_position r) (read_primitive r) (read_kind r))
        known k params metadata handler body r
    in
    match
      option_map
        (fun found =>
           (shares_reads known k (tests_of (read_position found) made above)
              params handler body found,
            found))
        (next_read params handler)
    with
    | Some (share, found) =>
      let r := with_reads_shared share params found in
      if loads_of_two_tags known k (tests_of (read_position r) made above)
           params handler body r
      then unchanged
      else
        let step :=
          match head_read params handler with
          | Some head =>
            if Pos.eqb (read_variable head) (read_variable r)
            then
              unbox_read
                (fun known =>
                   extend_args known (read_position head)
                     (read_primitive head))
                known k params metadata handler body
                (with_reads_shared share params head)
            else certain r
          | None => certain r
          end
        in
        match step with
        | Some (params', metadata, handler, body) =>
          unbox_reads fuel known k
            (if reads_test (read_primitive r)
             then (read_variable r, read_position r) :: made
             else made)
            ((length params,
              length params :: boxes_of (read_position r) above)
             :: above)
            params' metadata handler body
        | None => unchanged
        end
    | None => unchanged
    end
  end.

Fixpoint unbox_variants_knowing (known : facts) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind d (unbox_variants_knowing (facts_under_let known x d) body)
  | Let_static bindings body =>
    Let_static bindings
      (unbox_variants_knowing (facts_under_statics known bindings) body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    let handler :=
      unbox_variants_knowing (facts_under_params known params) handler
    in
    unbox_reads (lets_in handler) known k [] [] params metadata handler
      (unbox_variants_knowing known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (unbox_variants_knowing known body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition unbox_variant_params (e : expr) : expr :=
  unbox_variants_knowing no_facts e.

Definition unbox_variant_params_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (unbox_variant_params e).
